// Diagnostic: render the chat template exactly the way `llama-hidden-states` does and print
// the resulting prompt together with its token ids, so that the tokens fed to the model can
// be compared with what `llama-cli` feeds. Loads the vocab only (no weights, no GPU):
//
//   hs-prompt-dump model.gguf ["question"]
//
// The interesting part is `add_special=false` vs `true`: llama.cpp sets
// `inputs.add_bos = llama_vocab_get_add_bos()` and strips the leading bos text that the
// chat template prints, expecting the tokenizer to put the token back (llama-cli calls
// common_tokenize(..., true, true)). A prompt rendered without <bos> and tokenized with
// `false` starts with the first template token instead, which most models notice.
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static std::string escape_nl(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

static void print_ids(const char * label, const std::vector<llama_token> & ids, size_t limit) {
    printf("%s (%zu):", label, ids.size());
    for (size_t i = 0; i < ids.size() && i < limit; ++i) {
        printf(" %d", (int) ids[i]);
    }
    if (ids.size() > limit) {
        printf(" ...");
    }
    printf("\n");
}

static void dump(const common_chat_templates * tmpls, const llama_vocab * vocab,
                 const std::string & q, bool thinking) {
    common_chat_templates_inputs inputs;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = q;
    inputs.messages.push_back(msg);
    inputs.add_generation_prompt = true;
    inputs.enable_thinking       = thinking;

    const common_chat_params res = common_chat_templates_apply(tmpls, inputs);
#if __has_include("json.h")
    const std::string prompt = res.prompt.str();   // master: rendered prompt is common_chat_prompt
#else
    const std::string prompt = res.prompt;
#endif

    printf("--- thinking=%d  len=%zu ---\n%s\n", (int) thinking, prompt.size(), escape_nl(prompt).c_str());

    // does the rendered text still carry the bos token itself? then add_special=true would double it
    const std::string bos_piece = common_token_to_piece(vocab, llama_vocab_bos(vocab));
    printf("bos piece '%s' present in prompt text: %s\n", escape_nl(bos_piece).c_str(),
           prompt.find(bos_piece) != std::string::npos ? "yes" : "no");

    const std::vector<llama_token> toks = common_tokenize(vocab, prompt, false, true);
    print_ids("tokens add_special=false", toks, 8);
    const std::vector<llama_token> toks_s = common_tokenize(vocab, prompt, true, true);
    print_ids("tokens add_special=true ", toks_s, 8);
    printf("add_bos=%d  bos=%d  n_tokens=%zu/%zu\n", (int) llama_vocab_get_add_bos(vocab),
           (int) llama_vocab_bos(vocab), toks.size(), toks_s.size());

    printf("tail pieces:");
    const size_t from = toks.size() > 6 ? toks.size() - 6 : 0;
    for (size_t i = from; i < toks.size(); ++i) {
        printf(" [%s]", common_token_to_piece(vocab, toks[i]).c_str());
    }
    printf("\n\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [question]\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto tmpls = common_chat_templates_init(model, "");

    const std::string q = argc > 2 ? argv[2]
                                   : "Who wrote the play Hamlet? Answer with just the answer, no explanation.";
    dump(tmpls.get(), vocab, q, false);
    dump(tmpls.get(), vocab, q, true);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
