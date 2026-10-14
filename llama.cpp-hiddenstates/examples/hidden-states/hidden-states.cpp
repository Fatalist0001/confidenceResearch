#include "arg.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

static constexpr int HS_FLUSH_EVERY = 16;

//
// ---------------------------------------------------------------------------
// layer selection
// ---------------------------------------------------------------------------
//

// "all" | "step=K" | "4,8,12"
static bool parse_layers(const std::string & spec, int n_layer, std::vector<int> & out) {
    out.clear();

    if (spec == "all") {
        for (int i = 0; i < n_layer; ++i) {
            out.push_back(i);
        }
        return true;
    }

    if (spec.rfind("step=", 0) == 0) {
        int step = 0;
        try {
            step = std::stoi(spec.substr(5));
        } catch (...) {
            return false;
        }
        if (step <= 0) {
            return false;
        }
        for (int i = 0; i < n_layer; i += step) {
            out.push_back(i);
        }
        return !out.empty();
    }

    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int v = -1;
        try {
            v = std::stoi(item);
        } catch (...) {
            return false;
        }
        if (v < 0 || v >= n_layer) {
            return false;
        }
        out.push_back(v);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return !out.empty();
}

//
// ---------------------------------------------------------------------------
// eval callback (reads rows of l_out-N)
// ---------------------------------------------------------------------------
//

struct hs_state {
    bool enabled = false;

    int pos_a = -1;
    int pos_c = -1;
    int n_tokens = 0; // rows of the state pass, pos_c + 1
    bool want_a = true;

    int n_layer = 0;
    int n_embd  = 0;

    // layer -> row in the output buffer, -1 = not selected
    std::vector<int> layer_row;

    // reusable buffers, sized n_sel * n_embd
    std::vector<float> hs_c;
    std::vector<float> hs_a;

    // per-layer capture flags for the selected layers only
    std::vector<uint8_t> cap_c;
    std::vector<uint8_t> cap_a;
    int n_cap_c = 0;
    int n_cap_a = 0;

    bool dump_tensors = false;
};

static void reset_capture(hs_state & st) {
    std::fill(st.cap_c.begin(), st.cap_c.end(), (uint8_t) 0);
    std::fill(st.cap_a.begin(), st.cap_a.end(), (uint8_t) 0);
    st.n_cap_c = 0;
    st.n_cap_a = 0;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (hs_state *) user_data;

    if (ask) {
        if (st->dump_tensors) {
            if (strncmp(t->name, "l_out-", 6) == 0) {
                LOG_INF("tensor: %s shape=[%lld x %lld] type=%s\n", t->name,
                        (long long) t->ne[0], (long long) t->ne[1], ggml_type_name(t->type));
            } else {
                LOG_INF("tensor: %s\n", t->name);
            }
        }
        return st->enabled && strncmp(t->name, "l_out-", 6) == 0;
    }

    if (!st->enabled) {
        return true;
    }

    int layer = -1;
    if (sscanf(t->name, "l_out-%d", &layer) != 1) {
        return true;
    }
    if (layer < 0 || layer >= st->n_layer) {
        return true;
    }

    const int row = st->layer_row[layer];
    if (row < 0) {
        return true;
    }
    if (t->buffer == nullptr) {
        LOG_ERR("%s : tensor '%s' has no buffer\n", __func__, t->name);
        return true;
    }

    const int64_t D    = t->ne[0];
    const int64_t ntok = t->ne[1];

    if (D != st->n_embd) {
        LOG_ERR("%s : tensor '%s' has D=%lld, expected %d\n", __func__, t->name, (long long) D, st->n_embd);
        return true;
    }

    // l_out has one row per token; some backends keep only the rows that produced
    // logits, which are ordered as the logits were requested: A first, then C
    int64_t r_c = st->pos_c;
    int64_t r_a = st->pos_a;
    if (ntok <= st->pos_c) {
        r_c = st->pos_a == st->pos_c ? 0 : 1;
        r_a = 0;
    }
    const int64_t r_need = st->want_a ? std::max(r_c, r_a) : r_c;
    if (r_need >= ntok) {
        LOG_ERR("%s : tensor '%s' has %lld rows, need row %lld\n", __func__, t->name,
                (long long) ntok, (long long) r_need);
        return true;
    }

    const size_t el_size   = ggml_element_size(t);
    const size_t row_bytes = (size_t) D * el_size;

    auto read_row = [&](int r, float * dst) -> bool {
        std::vector<uint8_t> raw(row_bytes);
        ggml_backend_tensor_get(t, raw.data(), (size_t) r * t->nb[1], row_bytes);
        switch (t->type) {
            case GGML_TYPE_F32:
                {
                    const float * src = (const float *) raw.data();
                    for (int64_t i = 0; i < D; ++i) dst[i] = src[i];
                } break;
            case GGML_TYPE_F16:
                {
                    const ggml_fp16_t * src = (const ggml_fp16_t *) raw.data();
                    for (int64_t i = 0; i < D; ++i) dst[i] = ggml_fp16_to_fp32(src[i]);
                } break;
            case GGML_TYPE_BF16:
                {
                    const ggml_bf16_t * src = (const ggml_bf16_t *) raw.data();
                    for (int64_t i = 0; i < D; ++i) dst[i] = ggml_bf16_to_fp32(src[i]);
                } break;
            default:
                LOG_ERR("%s : unsupported type %s for tensor '%s'\n", __func__, ggml_type_name(t->type), t->name);
                return false;
        }
        return true;
    };

    {
        float * dst = st->hs_c.data() + (size_t) row * D;
        if (read_row((int) r_c, dst) && !st->cap_c[layer]) {
            st->cap_c[layer] = 1;
            st->n_cap_c++;
        }
    }

    if (st->want_a) {
        float * dst = st->hs_a.data() + (size_t) row * D;
        if (read_row((int) r_a, dst) && !st->cap_a[layer]) {
            st->cap_a[layer] = 1;
            st->n_cap_a++;
        }
    }

    return true;
}

//
// ---------------------------------------------------------------------------
// npy writers
// ---------------------------------------------------------------------------
//

// numpy .npy v1.0, C-order, little-endian float16, 2-D [L, D] (single-example mode)
static std::string npy_header_2d(int64_t L, int64_t D) {
    return "{'descr': '<f2', 'fortran_order': False, 'shape': (" +
        std::to_string(L) + ", " + std::to_string(D) + "), }";
}

// 3-D [N, L, D] (batch mode)
static std::string npy_header_3d(int64_t N, int64_t L, int64_t D) {
    return "{'descr': '<f2', 'fortran_order': False, 'shape': (" +
        std::to_string(N) + ", " + std::to_string(L) + ", " + std::to_string(D) + "), }";
}

static bool npy_write_header(std::ostream & f, std::string header, size_t * out_total) {
    const size_t base = 10;
    const size_t pad  = (64 - ((base + header.size() + 1) % 64)) % 64;
    header += std::string(pad, ' ');
    header += '\n';

    if (header.size() > 65535) {
        LOG_ERR("%s : npy header too large\n", __func__);
        return false;
    }

    static const uint8_t magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    f.write((const char *) magic, 6);

    static const uint8_t ver[2] = { 1, 0 };
    f.write((const char *) ver, 2);

    const uint16_t hlen = (uint16_t) header.size();
    f.write((const char *) &hlen, 2);
    f.write(header.data(), (std::streamsize) header.size());

    if (out_total) {
        *out_total = base + header.size();
    }

    return f.good();
}

static bool write_npy_f16(const fs::path & path, const float * data, int64_t L, int64_t D) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        LOG_ERR("%s : failed to open '%s' for writing\n", __func__, path.string().c_str());
        return false;
    }
    if (!npy_write_header(f, npy_header_2d(L, D), nullptr)) {
        return false;
    }

    const size_t n = (size_t) (L * D);
    std::vector<uint16_t> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = ggml_fp32_to_fp16(data[i]);
    }
    f.write((const char *) buf.data(), (std::streamsize) (n * sizeof(uint16_t)));

    return f.good();
}

// random-access float16 writer with a pre-sized [N, L, D] header
class npy_f16_writer {
    public:
        bool open(const fs::path & path, int64_t N, int64_t L, int64_t D, bool resume) {
            n_ = N;
            l_ = L;
            d_ = D;
            row_bytes_ = (size_t) (L * D) * sizeof(uint16_t);
            off_ = 0;

            if (!resume || !fs::exists(path)) {
                std::ofstream o(path, std::ios::binary | std::ios::trunc);
                if (!o.is_open()) {
                    LOG_ERR("%s : failed to open '%s' for writing\n", __func__, path.string().c_str());
                    return false;
                }
                if (!npy_write_header(o, npy_header_3d(N, L, D), &off_)) {
                    return false;
                }
                o.close();
            } else {
                // reuse the existing header as-is; data starts right after it
                std::ifstream i(path, std::ios::binary);
                if (!i.is_open()) {
                    LOG_ERR("%s : failed to open '%s'\n", __func__, path.string().c_str());
                    return false;
                }
                uint8_t hdr[10];
                i.read((char *) hdr, 10);
                if (i.gcount() != 10 || hdr[0] != 0x93) {
                    LOG_ERR("%s : '%s' is not a .npy file\n", __func__, path.string().c_str());
                    return false;
                }
                const uint16_t hlen = (uint16_t) hdr[8] | ((uint16_t) hdr[9] << 8);
                off_ = 10 + hlen;
            }

            f_.open(path, std::ios::binary | std::ios::in | std::ios::out);
            if (!f_.is_open()) {
                LOG_ERR("%s : failed to open '%s'\n", __func__, path.string().c_str());
                return false;
            }

            // make sure the file is big enough for all rows even if some are skipped
            {
                const std::streamoff need = (std::streamoff) (off_ + (uint64_t) n_ * row_bytes_);
                f_.seekp(0, std::ios::end);
                if (f_.tellp() < need) {
                    f_.seekp(need - 1);
                    char z = 0;
                    f_.put(z);
                }
                f_.seekp(0, std::ios::beg);
            }

            return true;
        }

        bool write_row(int64_t row, const float * data) {
            if (row < 0 || row >= n_) {
                LOG_ERR("%s : row %lld out of range [0, %lld)\n", __func__, (long long) row, (long long) n_);
                return false;
            }
            const size_t n = (size_t) (l_ * d_);
            buf_.resize(n);
            for (size_t i = 0; i < n; ++i) {
                buf_[i] = ggml_fp32_to_fp16(data[i]);
            }
            f_.seekp((std::streamoff) (off_ + (uint64_t) row * row_bytes_));
            f_.write((const char *) buf_.data(), (std::streamsize) (n * sizeof(uint16_t)));
            return f_.good();
        }

        bool flush() {
            f_.flush();
            return f_.good();
        }

        void close() {
            if (f_.is_open()) {
                f_.flush();
                f_.close();
            }
        }

    private:
        std::fstream f_;
        std::vector<uint16_t> buf_;
        int64_t n_ = 0;
        int64_t l_ = 0;
        int64_t d_ = 0;
        size_t  row_bytes_ = 0;
        size_t  off_ = 0;
};

//
// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

static bool check_finite(const std::vector<float> & v, const char * what) {
    size_t bad = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
            if (bad == 0) {
                LOG_ERR("%s : first non-finite value at flat index %zu\n", __func__, i);
            }
            bad++;
        }
    }
    if (bad > 0) {
        LOG_ERR("%s : %s has %zu non-finite values\n", __func__, what, bad);
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:   o += c;      break;
        }
    }
    return o;
}

// copies keys of `extra` into `dst` without overwriting anything already present
static void merge_extra(json & dst, const json & extra) {
    if (!extra.is_object()) {
        return;
    }
    for (auto it = extra.begin(); it != extra.end(); ++it) {
        if (!dst.contains(it.key())) {
            dst[it.key()] = it.value();
        }
    }
}

static std::string detok(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    std::string s;
    for (llama_token t : ids) {
        char buf[512];
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::vector<char> tmp((size_t) -n);
            llama_token_to_piece(vocab, t, tmp.data(), (int32_t) tmp.size(), 0, true);
            s.append(tmp.data(), tmp.size());
        } else {
            s.append(buf, (size_t) n);
        }
    }
    return s;
}

static int decode_ids(
        llama_context * ctx, const std::vector<llama_token> & ids, llama_pos pos0,
        const std::vector<int> & logits_at) {
    const int n = (int) ids.size();
    if (n <= 0) {
        return -1;
    }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i]      = ids[i];
        batch.pos[i]        = pos0 + i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = 0;
    }
    for (int i : logits_at) {
        if (i >= 0 && i < n) {
            batch.logits[i] = 1;
        }
    }
    batch.n_tokens = n;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        LOG_ERR("%s : llama_decode failed with code %d\n", __func__, rc);
    }
    return rc;
}

struct hs_opts {
    std::string batch_path;
    std::string out_dir;
    std::string layers      = "all";
    bool        generate    = false;
    bool        no_template = false;
    bool        thinking    = false; // false = --no-thinking (default for batch)
    int         max_new     = 64;
    int         max_len     = 1024;
};

//
// ---------------------------------------------------------------------------
// batch mode
// ---------------------------------------------------------------------------

struct hs_row {
    std::string id;
    std::string question;
    std::string answer;
    bool        has_answer = false;
    int64_t     row        = 0;
    json        extra;     // unknown keys of the input line, copied verbatim to meta.jsonl
};

static std::string render_prompt(
        const common_chat_templates * tmpls, const std::string & question, bool thinking) {
    common_chat_templates_inputs inputs;
    common_chat_msg user_msg;
    user_msg.role    = "user";
    user_msg.content = question;
    inputs.messages.push_back(user_msg);
    inputs.add_generation_prompt = true;
    inputs.enable_thinking       = thinking;

    const common_chat_params res = common_chat_templates_apply(tmpls, inputs);
    return res.prompt;
}

static int run_batch(common_params & params, hs_state & st, const hs_opts & o) {
    const std::string & batch_path  = o.batch_path;
    const std::string & out_dir_s   = o.out_dir;
    const bool          no_template = o.no_template;
    const bool          thinking    = o.thinking;
    const bool          do_generate = o.generate;
    const int           max_new     = o.max_new;
    const int           max_len     = o.max_len;
    const std::string & layers_spec = o.layers;

    if (batch_path.empty()) {
        LOG_ERR("%s : --batch requires an input .jsonl path\n", __func__);
        return 1;
    }
    if (out_dir_s.empty()) {
        LOG_ERR("%s : --batch requires --out <dir>\n", __func__);
        return 1;
    }

    // ---- load model + context once -------------------------------------
    params.warmup            = false;
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &st;
    params.n_ctx             = std::max(params.n_ctx, max_len);
    params.n_batch           = max_len;
    params.n_ubatch          = max_len;

    auto llama_init = common_init_from_params(params, /*model_only=*/true);

    llama_model * model = llama_init->model();
    if (model == nullptr) {
        LOG_ERR("%s : failed to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    st.n_layer = llama_model_n_layer(model);
    st.n_embd  = llama_model_n_embd(model);

    std::vector<int> layers;
    if (!parse_layers(layers_spec, st.n_layer, layers)) {
        LOG_ERR("%s : bad --layers value '%s' (n_layer=%d)\n", __func__, layers_spec.c_str(), st.n_layer);
        return 1;
    }
    st.layer_row.assign((size_t) st.n_layer, -1);
    for (size_t i = 0; i < layers.size(); ++i) {
        st.layer_row[(size_t) layers[i]] = (int) i;
    }
    st.cap_c.assign((size_t) st.n_layer, 0);
    st.cap_a.assign((size_t) st.n_layer, 0);

    const int64_t L_sel = (int64_t) layers.size();
    const int64_t D     = st.n_embd;

    st.hs_c.assign((size_t) L_sel * D, 0.0f);
    st.hs_a.assign((size_t) L_sel * D, 0.0f);
    st.want_a = true;
    reset_capture(st);

    common_chat_templates_ptr tmpls;
    if (!no_template) {
        tmpls = common_chat_templates_init(model, "");
        if (!tmpls) {
            LOG_ERR("%s : failed to init chat templates\n", __func__);
            return 1;
        }
    }

    llama_context * ctx = llama_init_from_model(model, common_context_params_to_llama(params));
    if (ctx == nullptr) {
        LOG_ERR("%s : failed to create context\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    LOG_INF("model: n_layer=%d n_embd=%d | selected layers=%lld | max_len=%d | template=%s | thinking=%s\n",
            st.n_layer, st.n_embd, (long long) L_sel, max_len, no_template ? "no" : "yes", thinking ? "yes" : "no");

    // ---- read input ----------------------------------------------------
    std::vector<hs_row> rows;
    {
        std::ifstream in(batch_path);
        if (!in.is_open()) {
            LOG_ERR("%s : failed to open '%s'\n", __func__, batch_path.c_str());
            return 1;
        }
        std::string line;
        int64_t idx = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            hs_row r;
            r.row = idx++;
            if (line.find_first_not_of(" \t") == std::string::npos) {
                r.id = "";
                r.question = "";
            } else {
                json j = json::parse(line, nullptr, false);
                if (j.is_discarded() || !j.is_object()) {
                    r.id = "";
                } else {
                    if (j.contains("id") && j["id"].is_string()) {
                        r.id = j["id"].get<std::string>();
                    }
                    if (j.contains("question") && j["question"].is_string()) {
                        r.question = j["question"].get<std::string>();
                    }
                    if (j.contains("answer") && j["answer"].is_string()) {
                        r.answer     = j["answer"].get<std::string>();
                        r.has_answer = true;
                    }
                    // pass every unknown key through to meta.jsonl unchanged
                    for (auto it = j.begin(); it != j.end(); ++it) {
                        const std::string & k = it.key();
                        if (k == "id" || k == "question" || k == "answer") {
                            continue;
                        }
                        r.extra[k] = it.value();
                    }
                }
            }
            rows.push_back(std::move(r));
        }
    }

    const int64_t N = (int64_t) rows.size();
    if (N == 0) {
        LOG_ERR("%s : '%s' is empty\n", __func__, batch_path.c_str());
        return 1;
    }

    // ---- outputs / resume ---------------------------------------------
    const fs::path out_dir(out_dir_s);
    const fs::path meta_path = out_dir / "meta.jsonl";
    std::error_code ec;
    fs::create_directories(out_dir, ec);

    std::unordered_set<std::string> done_ids;
    bool resume = fs::exists(meta_path);
    if (resume) {
        std::ifstream m(meta_path);
        std::string line;
        while (std::getline(m, line)) {
            json j = json::parse(line, nullptr, false);
            if (!j.is_discarded() && j.is_object() && j.contains("id") && j["id"].is_string()) {
                done_ids.insert(j["id"].get<std::string>());
            }
        }
        LOG_INF("resuming: %zu id(s) already present in %s\n", done_ids.size(), meta_path.string().c_str());
    }

    // ---- config.json ---------------------------------------------------
    {
        json cfg;
        cfg["model"]        = params.model.path;
        {
            char name_buf[512] = { 0 };
            llama_model_meta_val_str(model, "general.name", name_buf, sizeof(name_buf));
            cfg["model_name"] = name_buf;
        }
        cfg["n_layers"]           = st.n_layer;
        cfg["n_embd"]             = st.n_embd;
        cfg["layers"]             = layers;
        cfg["n_layers_selected"]  = L_sel;
        cfg["max_len"]            = max_len;
        cfg["max_new_tokens"]     = max_new;
        cfg["generate"]           = do_generate;
        cfg["template"]           = !no_template;
        cfg["thinking"]           = thinking;
        cfg["batch"]              = batch_path;
        cfg["n_examples"]         = N;
        cfg["point_definitions"] = {
            { "A", "last token of the prompt already rendered by the model chat template "
                   "with add_generation_prompt=true - the state from which the first answer token is emitted" },
            { "C", "last content token of the answer, EOG/replica-end tokens removed" },
        };
        cfg["llama_cpp_build"] = llama_build_number();
        cfg["llama_cpp_commit"] = llama_commit();
        cfg["ggml_version"]    = ggml_version();
        cfg["version"]         = 1;

        std::ofstream f(out_dir / "config.json");
        f << cfg.dump(2) << "\n";
    }

    // ---- output arrays -------------------------------------------------
    const fs::path path_a = out_dir / "hidden_a.npy";
    const fs::path path_c = out_dir / "hidden_c.npy";

    npy_f16_writer wr_a;
    npy_f16_writer wr_c;
    if (!wr_a.open(path_a, N, L_sel, D, resume) || !wr_c.open(path_c, N, L_sel, D, resume)) {
        return 1;
    }

    std::ofstream meta(meta_path, std::ios::app);
    if (!meta.is_open()) {
        LOG_ERR("%s : failed to open '%s'\n", __func__, meta_path.string().c_str());
        return 1;
    }

    // ---- process -------------------------------------------------------
    const auto t_start = std::chrono::steady_clock::now();
    int64_t done   = 0;
    int64_t n_ok   = 0;
    int64_t n_skip = 0;
    int64_t n_run  = 0; // newly processed (for speed)
    int64_t n_trunc = 0;
    int64_t since_flush = 0;
    int64_t n_gen_rows  = 0; // rows that went through generation
    int64_t n_retok     = 0; // .. whose text re-tokenizes back to different ids
    bool    logged_prompt = false;

    auto write_meta = [&](const json & j) {
        meta << j.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
    };

    for (const hs_row & r : rows) {
        done++;

        if (!r.id.empty() && done_ids.count(r.id)) {
            continue;
        }

        st.enabled = false;
        reset_capture(st);

        auto skip = [&](const std::string & reason) {
            json j;
            j["id"]        = r.id;
            j["row"]       = r.row;
            j["n_tokens"]  = 0;
            j["pos_a"]     = -1;
            j["pos_c"]     = -1;
            j["answer"]    = r.answer;
            j["generated"] = false;
            j["truncated"] = false;
            j["logprobs"]  = nullptr;
            j["entropies"] = nullptr;
            j["status"]    = "skipped";
            j["reason"]    = reason;
            merge_extra(j, r.extra);
            write_meta(j);
            n_skip++;
            n_run++;
            LOG_WRN("row %lld id=%s : skipped (%s)\n", (long long) r.row, r.id.c_str(), reason.c_str());
        };

        if (r.question.empty() && !r.has_answer && r.id.empty()) {
            skip("empty or invalid json line");
            continue;
        }
        if (r.question.empty()) {
            skip("empty question");
            continue;
        }

        // 1. prompt text
        std::string prompt_text;
        if (no_template) {
            prompt_text = r.question;
        } else {
            try {
                prompt_text = render_prompt(tmpls.get(), r.question, thinking);
            } catch (const std::exception & e) {
                skip(std::string("chat template failed: ") + e.what());
                continue;
            }
        }

        if (!logged_prompt) {
            logged_prompt = true;
            LOG_INF("rendered prompt[0] (row %lld):\n%s\n", (long long) r.row, prompt_text.c_str());
        }

        std::vector<llama_token> tok_q = common_tokenize(vocab, prompt_text, no_template ? add_bos : false, true);
        if (tok_q.empty()) {
            skip("prompt produced no tokens");
            continue;
        }

        std::string answer_text = r.answer;
        bool generated = false;
        bool truncated = false;
        std::vector<llama_token> gen;
        std::vector<float> logprobs;
        std::vector<float> entropies;

        // 2. generation (only when the answer is missing)
        if (!r.has_answer && do_generate) {
            if ((int64_t) tok_q.size() >= max_len) {
                skip("no room for generation (prompt >= max_len)");
                continue;
            }

            llama_memory_clear(mem, true);
            st.enabled = false;

            std::string stop_reason = "max_new_tokens";

            {
                const int n_prompt = (int) tok_q.size();
                int logits_idx = n_prompt - 1;

                if (decode_ids(ctx, tok_q, 0, { logits_idx }) != 0) {
                    skip("decode failed during generation prompt");
                    continue;
                }

                for (int i = 0; i < max_new; ++i) {
                    const float * logits = llama_get_logits_ith(ctx, logits_idx);
                    if (logits == nullptr) {
                        stop_reason = "no logits";
                        break;
                    }
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    llama_token best = 0;
                    float best_v = logits[0];
                    for (int v = 1; v < n_vocab; ++v) {
                        if (logits[v] > best_v) {
                            best_v = logits[v];
                            best   = (llama_token) v;
                        }
                    }

                    if (llama_vocab_is_eog(vocab, best)) {
                        stop_reason = "eog";
                        break;
                    }

                    gen.push_back(best);

                    // baseline statistics of the plain softmax over the raw logits
                    {
                        const double best_logit = (double) best_v;
                        double z = 0.0;
                        for (int v = 0; v < n_vocab; ++v) {
                            z += std::exp((double) logits[v] - best_logit);
                        }
                        const double log_z = best_logit + std::log(z);
                        double h = 0.0;
                        for (int v = 0; v < n_vocab; ++v) {
                            const double p = std::exp((double) logits[v] - log_z);
                            if (p > 0.0) {
                                h -= p * ((double) logits[v] - log_z);
                            }
                        }
                        logprobs.push_back((float) (best_logit - log_z));
                        entropies.push_back((float) h);
                    }

                    if ((int64_t) (tok_q.size() + gen.size()) >= max_len) {
                        stop_reason = "max_len";
                        break;
                    }

                    const int pos = n_prompt + (int) gen.size() - 1;
                    if (decode_ids(ctx, { best }, pos, { 0 }) != 0) {
                        stop_reason = "decode failed";
                        break;
                    }
                    logits_idx = 0;
                }
            }

            answer_text = detok(vocab, gen);
            generated   = true;
            // the answer is incomplete when the run stopped on the token/length
            // budget instead of the model emitting an EOG token
            truncated   = stop_reason != "eog";

            LOG_INF("  generated %zu token(s), stop=%s\n", gen.size(), stop_reason.c_str());
        }

        // 3. answer tokens (EOG tails stripped)
        std::vector<llama_token> tok_a;
        if (generated) {
            // ids produced by generation are used as-is for the state pass
            tok_a = gen;
            n_gen_rows++;

            std::vector<llama_token> re = common_tokenize(vocab, answer_text, false, true);
            while (!re.empty() && llama_vocab_is_eog(vocab, re.back())) {
                re.pop_back();
            }
            if (re != gen) {
                n_retok++;
            }
        } else if (!answer_text.empty()) {
            tok_a = common_tokenize(vocab, answer_text, false, true);
            while (!tok_a.empty() && llama_vocab_is_eog(vocab, tok_a.back())) {
                tok_a.pop_back();
            }
        }

        std::vector<llama_token> ids = tok_q;
        ids.insert(ids.end(), tok_a.begin(), tok_a.end());

        const int n_prompt = (int) tok_q.size();
        const int n_tokens = (int) ids.size();

        const int pos_a = n_prompt - 1;
        const int pos_c = n_tokens - 1;

        if (n_tokens > max_len) {
            skip("too long: " + std::to_string(n_tokens) + " tokens > max_len " + std::to_string(max_len));
            continue;
        }

        // 4. state pass over "prompt + answer"
        llama_memory_clear(mem, true);

        st.enabled  = true;
        st.pos_a    = pos_a;
        st.pos_c    = pos_c;
        st.n_tokens = n_tokens;
        reset_capture(st);

        const int logits_a = pos_a;
        const int logits_c = pos_c;
        const std::vector<int> logits_at = logits_a == logits_c ? std::vector<int>{ logits_c } : std::vector<int>{ logits_a, logits_c };
        if (decode_ids(ctx, ids, 0, logits_at) != 0) {
            st.enabled = false;
            skip("decode failed");
            continue;
        }

        st.enabled = false;

        const int cap = st.want_a ? std::min(st.n_cap_c, st.n_cap_a) : st.n_cap_c;
        if (cap != (int) L_sel) {
            skip("incomplete capture: " + std::to_string(cap) + "/" + std::to_string(L_sel));
            continue;
        }

        if (!check_finite(st.hs_c, "hs_c") || !check_finite(st.hs_a, "hs_a")) {
            skip("failed to capture hidden states");
            continue;
        }

        if (!wr_c.write_row(r.row, st.hs_c.data()) || !wr_a.write_row(r.row, st.hs_a.data())) {
            LOG_ERR("%s : failed to write row %lld\n", __func__, (long long) r.row);
            return 1;
        }

        {
            json j;
            j["id"]        = r.id;
            j["row"]       = r.row;
            j["n_tokens"]  = n_tokens;
            j["pos_a"]     = pos_a;
            j["pos_c"]     = pos_c;
            j["answer"]    = answer_text;
            j["generated"] = generated;
            j["truncated"] = truncated;
            j["status"]    = "ok";
            if (generated) {
                j["logprobs"]  = logprobs;
                j["entropies"] = entropies;
            } else {
                j["logprobs"]  = nullptr;
                j["entropies"] = nullptr;
            }
            merge_extra(j, r.extra);
            write_meta(j);
        }

        n_ok++;
        n_run++;
        if (truncated) {
            n_trunc++;
        }

        if (++since_flush >= HS_FLUSH_EVERY) {
            since_flush = 0;
            wr_a.flush();
            wr_c.flush();
            meta.flush();
        }

        // progress
        {
            const auto t_now = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t_now - t_start).count();
            const double sps = sec > 0.0 ? (double) n_run / sec : 0.0;
            fprintf(stderr, "\r[%lld/%lld] ok=%lld skipped=%lld truncated=%lld  %.2f ex/s   ",
                    (long long) done, (long long) N, (long long) n_ok, (long long) n_skip, (long long) n_trunc, sps);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n");

    wr_a.close();
    wr_c.close();
    meta.flush();
    meta.close();

    const auto t_end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t_end - t_start).count();

    LOG_INF("\n");
    LOG_INF("done: %lld/%lld  ok=%lld skipped=%lld truncated=%lld  in %.1f s (%.2f ex/s)\n",
            (long long) done, (long long) N, (long long) n_ok, (long long) n_skip, (long long) n_trunc, sec, sec > 0.0 ? (double) n_run / sec : 0.0);
    if (n_gen_rows > 0) {
        LOG_INF("retokenize mismatch: %lld/%lld (%.1f%%)\n", (long long) n_retok, (long long) n_gen_rows,
                100.0 * (double) n_retok / (double) n_gen_rows);
    }
    LOG_INF("wrote %s  shape=[%lld, %lld, %lld]\n", path_c.string().c_str(), (long long) N, (long long) L_sel, (long long) D);
    LOG_INF("wrote %s  shape=[%lld, %lld, %lld]\n", path_a.string().c_str(), (long long) N, (long long) L_sel, (long long) D);
    LOG_INF("wrote %s\n", (out_dir / "meta.jsonl").string().c_str());
    LOG_INF("wrote %s\n", (out_dir / "config.json").string().c_str());

    llama_free(ctx);
    llama_init.reset();
    llama_backend_free();

    return 0;
}

//
// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // custom flags are stripped before common_params_parse() (it rejects unknown args)
    std::string opt_answer;
    std::string opt_out;
    std::string opt_point_a;
    bool        opt_dump_tensors = false;

    std::string opt_batch;
    std::string opt_layers = "all";
    bool        opt_generate     = false;
    bool        opt_no_template  = false;
    bool        opt_thinking     = false; // --no-thinking is the batch default
    int         opt_max_new      = 64;
    int         opt_max_len      = 1024;

    std::vector<char *> cargv;
    cargv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                LOG_ERR("%s : missing value for '%s'\n", __func__, name);
                exit(1);
            }
            return argv[++i];
        };
        auto next_int = [&](const char * name) -> int {
            const std::string v = next(name);
            try {
                return std::stoi(v);
            } catch (...) {
                LOG_ERR("%s : bad value for '%s': '%s'\n", __func__, name, v.c_str());
                exit(1);
            }
        };

        if (arg == "--answer")               { opt_answer       = next("--answer");       continue; }
        if (arg.rfind("--answer=", 0) == 0)  { opt_answer       = arg.substr(9);          continue; }
        if (arg == "--out")                  { opt_out          = next("--out");          continue; }
        if (arg.rfind("--out=", 0) == 0)     { opt_out          = arg.substr(6);          continue; }
        if (arg == "--point-a")              { opt_point_a      = next("--point-a");      continue; }
        if (arg.rfind("--point-a=", 0) == 0) { opt_point_a      = arg.substr(10);         continue; }
        if (arg == "--dump-tensors")         { opt_dump_tensors = true;                   continue; }

        if (arg == "--batch")                { opt_batch        = next("--batch");        continue; }
        if (arg.rfind("--batch=", 0) == 0)   { opt_batch        = arg.substr(8);          continue; }
        if (arg == "--layers")               { opt_layers       = next("--layers");       continue; }
        if (arg.rfind("--layers=", 0) == 0)  { opt_layers       = arg.substr(9);          continue; }
        if (arg == "--generate")             { opt_generate     = true;                   continue; }
        if (arg == "--no-template")          { opt_no_template  = true;                   continue; }
        if (arg == "--no-thinking")          { opt_thinking     = false;                  continue; }
        if (arg == "--thinking")             { opt_thinking     = true;                   continue; }
        if (arg == "--max-new-tokens")       { opt_max_new      = next_int("--max-new-tokens"); continue; }
        if (arg.rfind("--max-new-tokens=", 0) == 0) { opt_max_new = std::stoi(arg.substr(16)); continue; }
        if (arg == "--max-len")              { opt_max_len      = next_int("--max-len");  continue; }
        if (arg.rfind("--max-len=", 0) == 0) { opt_max_len      = std::stoi(arg.substr(10)); continue; }

        cargv.push_back(argv[i]);
    }

    common_params params;

    hs_opts opts;
    opts.batch_path  = opt_batch;
    opts.out_dir     = opt_out;
    opts.layers      = opt_layers;
    opts.generate    = opt_generate;
    opts.no_template = opt_no_template;
    opts.thinking    = opt_thinking;
    opts.max_new     = opt_max_new;
    opts.max_len     = opt_max_len;

    common_init();

    if (!common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (!opt_batch.empty()) {
        llama_backend_init();
        llama_numa_init(params.numa);

        hs_state st;
        st.dump_tensors = opt_dump_tensors;

        return run_batch(params, st, opts);
    }

    // ---- single-example mode -------------------------------------------

    int pos_a_user = -1;
    if (!opt_point_a.empty()) {
        try {
            pos_a_user = std::stoi(opt_point_a);
        } catch (...) {
            LOG_ERR("%s : bad --point-a value '%s' (expected integer token index)\n", __func__, opt_point_a.c_str());
            return 1;
        }
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s : empty prompt - provide one with '-p'\n", __func__);
        return 1;
    }

    hs_state st;
    st.dump_tensors = opt_dump_tensors;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.warmup            = false;
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &st;

    if (params.n_ctx <= 0) {
        params.n_ctx = 4096;
    }
    params.n_batch  = std::max(params.n_batch,  1024);
    params.n_ubatch = std::max(params.n_ubatch, 1024);

    auto llama_init = common_init_from_params(params, /*model_only=*/true);

    llama_model * model = llama_init->model();
    if (model == nullptr) {
        LOG_ERR("%s : failed to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tok_q = common_tokenize(vocab, params.prompt, add_bos, true);
    std::vector<llama_token> tok_a = common_tokenize(vocab, opt_answer,   false,   true);

    if (tok_q.empty()) {
        LOG_ERR("%s : the prompt produced no tokens - (try to provide a prompt with '-p')\n", __func__);
        return 1;
    }

    std::vector<llama_token> ids = tok_q;
    ids.insert(ids.end(), tok_a.begin(), tok_a.end());

    const int n_prompt = (int) tok_q.size();
    const int n_answer = (int) tok_a.size();
    const int n_tokens = (int) ids.size();

    const int pos_c = n_tokens - 1;
    const int pos_a = pos_a_user >= 0 ? pos_a_user : n_prompt - 1;

    if (pos_a >= n_prompt) {
        LOG_ERR("%s : --point-a %d is out of range, the question has %d tokens (0..%d)\n",
                __func__, pos_a, n_prompt, n_prompt - 1);
        return 1;
    }

    params.n_batch  = std::max(params.n_batch,  n_tokens);
    params.n_ubatch = std::max(params.n_ubatch, n_tokens);
    params.n_ctx    = std::max(params.n_ctx,    n_tokens + 256);

    LOG_INF("tokens: prompt=%d answer=%d total=%d | pos_a=%d pos_c=%d | n_batch=%d n_ubatch=%d n_ctx=%d\n",
            n_prompt, n_answer, n_tokens, pos_a, pos_c, params.n_batch, params.n_ubatch, params.n_ctx);

    llama_context * ctx = llama_init_from_model(model, common_context_params_to_llama(params));
    if (ctx == nullptr) {
        LOG_ERR("%s : failed to create context\n", __func__);
        return 1;
    }

    st.n_layer   = llama_model_n_layer(model);
    st.n_embd    = llama_model_n_embd(model);
    st.pos_a     = pos_a;
    st.pos_c     = pos_c;
    st.n_tokens  = n_tokens;
    st.want_a    = true;
    st.layer_row.assign((size_t) st.n_layer, -1);
    for (int i = 0; i < st.n_layer; ++i) {
        st.layer_row[(size_t) i] = i;
    }
    st.hs_c.assign((size_t) st.n_layer * st.n_embd, 0.0f);
    st.hs_a.assign((size_t) st.n_layer * st.n_embd, 0.0f);
    st.cap_c.assign((size_t) st.n_layer, 0);
    st.cap_a.assign((size_t) st.n_layer, 0);
    reset_capture(st);
    st.enabled = true;

    LOG_INF("model: n_layer=%d n_embd=%d\n", st.n_layer, st.n_embd);

    {
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i]      = ids[i];
            batch.pos[i]        = i;
            batch.n_seq_id[i]   = 1;
            batch.seq_id[i][0]  = 0;
            batch.logits[i]     = (i == pos_c) || (i == pos_a);
        }
        batch.n_tokens = n_tokens;

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            LOG_ERR("%s : llama_decode failed with code %d\n", __func__, rc);
            llama_free(ctx);
            return 1;
        }
    }

    llama_synchronize(ctx);

    st.enabled = false;

    {
        const int cap = st.want_a ? std::min(st.n_cap_c, st.n_cap_a) : st.n_cap_c;
        if (cap != st.n_layer) {
            LOG_ERR("%s : incomplete capture: %d/%d\n", __func__, cap, st.n_layer);
            llama_free(ctx);
            return 1;
        }
    }

    if (!check_finite(st.hs_c, "hs_c")) { llama_free(ctx); return 1; }
    if (!check_finite(st.hs_a, "hs_a")) { llama_free(ctx); return 1; }

    fs::path out = opt_out.empty() ? fs::path(".") : fs::path(opt_out);

    fs::path path_c;
    fs::path path_a;
    fs::path path_json;
    if (out.extension() == ".npy") {
        fs::path base = out;
        base.replace_extension();
        path_c    = out;
        path_a    = fs::path(base.string() + "_a.npy");
        path_json = fs::path(base.string() + ".json");
    } else {
        std::error_code ec;
        fs::create_directories(out, ec);
        path_c    = out / "hidden_c.npy";
        path_a    = out / "hidden_a.npy";
        path_json = out / "hidden_states.json";
    }

    const int64_t L = st.n_layer;
    const int64_t D = st.n_embd;

    if (!write_npy_f16(path_c, st.hs_c.data(), L, D)) { llama_free(ctx); return 1; }
    if (!write_npy_f16(path_a, st.hs_a.data(), L, D)) { llama_free(ctx); return 1; }

    char name_buf[512] = { 0 };
    llama_model_meta_val_str(model, "general.name", name_buf, sizeof(name_buf));

    {
        std::ofstream f(path_json);
        if (!f.is_open()) {
            LOG_ERR("%s : failed to open '%s' for writing\n", __func__, path_json.string().c_str());
            llama_free(ctx);
            return 1;
        }
        f << "{\n";
        f << "  \"model\": \"" << json_escape(name_buf) << "\",\n";
        f << "  \"model_path\": \"" << json_escape(params.model.path) << "\",\n";
        f << "  \"n_layers\": " << L << ",\n";
        f << "  \"n_embd\": " << D << ",\n";
        f << "  \"n_tokens\": " << n_tokens << ",\n";
        f << "  \"n_prompt_tokens\": " << n_prompt << ",\n";
        f << "  \"n_answer_tokens\": " << n_answer << ",\n";
        f << "  \"pos_a\": " << pos_a << ",\n";
        f << "  \"pos_c\": " << pos_c << ",\n";
        f << "  \"files\": {\n";
        f << "    \"c\": \"" << json_escape(path_c.string()) << "\",\n";
        f << "    \"a\": \"" << json_escape(path_a.string()) << "\"\n";
        f << "  }\n";
        f << "}\n";
    }

    LOG_INF("\n");
    LOG_INF("wrote %s  shape=[%lld, %lld]  (position C=%d)\n", path_c.string().c_str(), (long long) L, (long long) D, pos_c);
    LOG_INF("wrote %s  shape=[%lld, %lld]  (position A=%d)\n", path_a.string().c_str(), (long long) L, (long long) D, pos_a);
    LOG_INF("wrote %s\n", path_json.string().c_str());

    llama_free(ctx);
    llama_init.reset();
    llama_backend_free();

    return 0;
}
