# qwen3.cu

`qwen3.cu` is a **single-file, pure CUDA C implementation** for running inference on the Qwen3 model with no external libraries, no dependencies. It’s a follow-up to my earlier weekend project, [qwen3.c](https://github.com/...), inspired by Andrej Karpathy’s [`llama2.c`](https://github.com/karpathy/llama2.c). Everything’s packed into one file from tokenization all the way to CUDA kernels, staying true to the spirit of minimalism.

This implementation runs the Qwen3 0.6B model, a small but capable model. I'm using **full-precision GGUF** here, chosen for its clarity and to help others learn its ways. Also, It’s fully self-contained, so there’s no need for any format conversion out of the box. Most GGUF models are quantized to 8-bit or lower, but for this project, you’ll want to use the FP32 version which you can download as below. Or, if you make it work from the BF16 weights, you can convert them using the included `convert_hf_to_gguf_ordered.py` script; I've made sure the layers are ordered numerically so everything aligns correctly.

Even though GGUF files already include a binary tokenizer, this project reads vocab and merges from plain `.txt` files. It keeps things more transparent and easier to follow. Tokenization and detokenization overhead is negligible compared to the forward pass, so it doesn’t really impact TTS.

It also supports multi-turn conversation out of the box, and native support for Qwen3’s reasoning mode. For reference, there’s also a cuBLAS version included. It’s roughly 2x faster for now, but I’ll probably try to narrow that gap in the future. I’ll add more explanation on the code later.

## Quick Start

```sh
# Clone this repo
git clone https://github.com/gigit0000/qwen3.cu.git
cd qwen3.cu

# Download FP32 model from Hugging Face
git clone https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32
mv Qwen3-0.6B-GGUF-FP32/Qwen3-0.6B-FP32.gguf ./

# Compile and run
make runcu
./runcu Qwen3-0.6B-FP32.gguf
```

## Faster Inference
Use cuBLAS:
```sh
# Compile and run
make runcublas
./runcublas Qwen3-0.6B-FP32.gguf 
```
## Description

You can enable reasoning (-k 0) or multi-turn (-m 0):
```
./runcu Qwen3-0.6B-FP32.gguf -k 0 -m 0 
```

If you want to extract text files (vocab.txt, merges.txt and header.txt) on your own, you can use the scripts:
```sh
# tokenizer - vocab.txt and merges.txt
python extract_v_m.py Qwen3-0.6B-FP32.gguf

```

### Inference Examples

Multi-turn Conversation with the option m
```
# ./runcu Qwen3-0.6B-FP32.gguf -m 1 -k 0
Multi-turn = on, thinKing = off, Temperature = 0.60, top-P = 0.95
Press Enter to exit the chat
Enter system prompt (or Enter to skip): Tell me in one sentence
Q: Where is the best spot in Paris?
A: The best spot in Paris is the Eiffel Tower.
Q: What about the second-best spot?
A: The second-best spot in Paris is the Louvre Museum.
```

Reasoning with the option k
```
# ./runcu Qwen3-0.6B-FP32.gguf -k 1
Multi-turn = off, thinKing = on, Temperature = 0.60, top-P = 0.95
Press Enter to exit the chat
Enter system prompt (or Enter to skip): 
Q: Why do stars shine? Give me a quick answer!
A: <think>
Okay, the user is asking why stars shine. Let me start by recalling what I know about stars. Stars are luminous objects that emit light. So, the main reason they shine is because they produce light through nuclear fusion.

Wait, but I should make sure. Stars form from clouds of gas and dust in space. When these clouds cool, they start fusing hydrogen into helium, which releases energy. This energy is what we see as light. So the process is nuclear fusion of hydrogen into helium, which gives off energy.

I should also mention that the energy from stars is what we perceive as light. Maybe add that this light travels through space and we see it on Earth. But the question is why they shine, so the answer should focus on the energy production.

I need to keep it simple and concise. The user probably wants a quick answer, so no need for too much detail. Let me check if there's any other reason, but I think that's the main one. Alright, I think that's it.
</think>

Stars shine because they produce light through nuclear fusion of hydrogen into helium in their cores. This energy is then released as visible light, giving them their luminous glow.
```
You can enable and monitor TPS with the r option:
```
./runcu Qwen3-0.6B-FP32.gguf -r 1 
Multi-turn = off, thinKing = off, tps(R) = on, Temperature = 0.60, top-P = 0.95
Press Enter to exit the chat
Enter system prompt (or Enter to skip): You name is Tom.
Q: What is your name?
A: My name is Tom.
tok/s: 34.482759
```

## (Maybe) TODO
- [ ] Kernel optimization
- [ ] CUTLASS version
- [ ] KV cache for multi-turn conversations

## Acknoledgement
- Inspired and baselined from Andrej Kapathy's [llama2.c](https://github.com/karpathy/llama2.c)
- Most kernels and CUDA ports were originally adopted from @rogerallen's great repo [llama2.cu](https://github.com/rogerallen/)
- Based on my qwen3.c [repo](https://github.com/gigit0000/qwen3.c/)
- FGPF

## License
MIT






