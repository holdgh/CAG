import torch
import argparse
import os
import cag.dataset as cagds
import cag.similarity as cagsim
from time import time
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import logging 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN not found")


"""Hugging Face Llama model"""

global model_name, model, tokenizer
global rand_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


"""KV Cache test"""
# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


def generate(
    model,
    input_ids: torch.Tensor,  # 问题向量
    past_key_values,  # 知识的注意力键值信息，又称为知识的kv缓存
    max_new_tokens: int = 300
) -> torch.Tensor:  # rag的理念：基于问题检索数据，将数据和问题一同放入提示词中给到大模型，生成回答。cag的理念：将知识通过大模型生成知识对应的kv缓存【放入文件中】，直接将知识对应的kv缓存和问题给到大模型，生成回答。
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        past_key_values: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, 
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            past_key_values = outputs.past_key_values  # 生成一个token后，更新知识的kv缓存

            output_ids = torch.cat([output_ids, next_token], dim=1)

            # if next_token.item() in model.config.eos_token_id:
            if next_token.item() == model.config.eos_token_id:
                break
    return output_ids[:, origin_ids.shape[-1]:]


def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str,
) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)  # prompt在这里是除了问答对话提示词中问题外的所有内容，也即上下文知识。这里将知识进行分词，并放入cpu中，返回分词后的tensor
    '''
    众所周知，Transformers库实现中为节约计算时间，会将之前计算的注意力的键与值缓存下来，从而起到加速解码的作用，对应的键与值储存在输出的past_key_values中。在实际应用中，如果已有之前文本的past_key_values，想要继续进行文本生成，则可以只输入新的input_ids与之前的past_key_values，就可以继续进行生成。
    '''
    past_key_values = DynamicCache()  # 初始化一个动态缓存对象，大模型
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    return outputs.past_key_values  # 将知识对应的注意力键值信息输出


def write_kv_cache(kv: DynamicCache, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def read_kv_cache(path: str) -> DynamicCache | None:
    """
    Read the KV Cache from a file. If the cache file is invalid or empty, return None.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        kv = torch.load(path, weights_only=True)
        return kv
    else:
        # Regenerate cache if it doesn't exist or is too small
        return None


def prepare_kvcache(documents, filepath: str = "./data_cache/cache_knowledges.pt", answer_instruction: str | None = None):
    # Prepare the knowledges kvcache

    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """  # 将选中的5篇文章内容，连同answer_instruction一并放入knowledges【下称“知识”】中
    # Get the knowledge cache
    t1 = time()
    kv = preprocess_knowledge(model, tokenizer, knowledges)  # 利用大模型处理知识，得到知识的注意力键值信息
    print("kvlen: ", kv.key_cache[0].shape[-2])
    write_kv_cache(kv, filepath)  # 将知识的注意力键值信息写入文件
    t2 = time()
    logger.info(f"KV cache prepared in {t2 - t1:.2f} seconds.")
    return kv, t2 - t1


def kvcache_test(args: argparse.Namespace):
    answer_instruction = "Answer the question with a super short answer."
    text_list, dataset = cagds.get(args.dataset, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)  # 获取文章列表和问答对列表

    kvcache_path = "./data_cache/cache_knowledges.pt"  # 声明缓存数据路径

    knowledges = '\n\n\n\n\n\n'.join(text_list)  # 将文章列表【若有5篇文章，则文章列表有10个元素，5个标题和5个文章】拼接为整个字符串，用6个换行符分隔
    knowledge_cache, prepare_time = prepare_kvcache(knowledges, filepath=kvcache_path, answer_instruction=answer_instruction)  # 利用大模型获取知识上下文的注意力键值信息
    kv_len = knowledge_cache.key_cache[0].shape[-2]
    print(f"KVcache prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"KVcache prepared in {prepare_time} seconds\n")

    results = {
        "cache_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }
    # 将问答集转换为列表
    dataset = list(dataset)  # Convert the dataset to a list

    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion is not None else len(dataset)  # 设置最大问题数量
    # Retrieve the knowledge from the vector database
    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):  # 处理前max_questions个问答对
        torch.cuda.empty_cache()  # 清空gpu缓存
        torch.cuda.ipc_collect()  # 收集gpu缓存

        # Read the knowledge cache from the cache file
        cache_t1 = time()
        # if args.kvcache == "file":
        #     knowledge_cache = read_kv_cache(kvcache_path)

        # Not a good idea to use this method, as it will consume a lot of memory
        # if args.kvcache == "variable":
        #     knowledge_cache = documents_cache
        cache_t2 = time()

        # Generate Response for the question
        knowledges = '\n\n\n'.join(text_list)  # 将文章列表拼接为知识

        if args.usePrompt:
            prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {knowledges}
    ------------------------------------------------
    {answer_instruction}
    Question:
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
            generate_t1 = time()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, DynamicCache()) 
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
            generate_t2 = time()
        else:  # 参数中usePrompt为False
            # 此时prompt仅为问题
            prompt = f"""
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
            generate_t1 = time()
            clean_up(knowledge_cache, kv_len)  # 将知识的注意力键值信息【kvcache】截断为特定长度
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)  # 将问题转化为大模型的输入tensor
            output = generate(model, input_ids, knowledge_cache)  # 利用知识的注意力键值信息【已经计算好的知识上下文的kvcache】回答问题
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)  # 解码输出结果，转换为文字
            generate_t2 = time()

        # print("D: ", knowledges)
        print("Q: ", question)
        print("A: ", generated_text)
 
        # Evaluate bert-score similarity
        similarity = cagsim.bert(generated_text, ground_truth)

        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},",
              f"cache time: {cache_t2 - cache_t1},",
              f"generate time: {generate_t2 - generate_t1}")
        with open(args.output, "a") as f:
            f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t cache time: {cache_t2 - cache_t1},\t generate time: {generate_t2 - generate_t1}\n")

        results["prompts"].append(question)
        results["responses"].append(generated_text)
        results["cache_time"].append(cache_t2 - cache_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)

        with open(args.output, "a") as f:
            f.write(f"[{id}]: [Cumulative]: "
                    + f"Semantic Similarity: {round(sum(results['similarity']) / (len(results['similarity'])) , 5)},"
                    + f"\t cache time: {sum(results['cache_time']) / (len(results['cache_time'])) },"
                    + f"\t generate time: {sum(results['generate_time']) / (len(results['generate_time'])) }\n")

    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Result for {args.output}\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}\n")


# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
)


def load_quantized_model(model_name, hf_token=None):
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name,
    #     token=hf_token
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # Load model with quantization
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    #     device_map="auto",          # Automatically choose best device
    #     trust_remote_code=True,     # Required for some models
    #     token=hf_token
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True
    )

    return tokenizer, model


if __name__ == "__main__":

    '''
    --kvcache file 缓存方式【文件 or 变量】
    --dataset "squad-train" 数据集
    --similarity bertscore 相似度
    --maxKnowledge 5 最大知识数量【？】
    --maxParagraph 100 最大段落数量【长度？】
    --maxQuestion 1000 最大问题数量【长度？】
    --modelname "meta-llama/Llama-3.1-8B-Instruct" llm 
    --randomSeed 0 随机种子
    --output "./result_kvcache.txt" 输出结果到文件
    '''
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    # parser.add_argument('--kvcache', choices=['file', 'variable'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--modelname', required=False, default="meta-llama/Llama-3.2-1B-Instruct", type=str, help='Model name to use')
    parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
    parser.add_argument('--kvcache', choices=['file'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=None, type=int, help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', required=False, default=None, type=int, help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None, type=int, help='Maximum number of paragraph to use')
    parser.add_argument('--usePrompt', default=False, action="store_true", help='Do not use cache')
    parser.add_argument('--dataset', required=True, help='Dataset to use (kis, kis_sample, squad-dev, squad-train)',
                        choices=['kis', 'kis_sample',
                                 'squad-dev', 'squad-train',
                                 'hotpotqa-dev',  'hotpotqa-train', 'hotpotqa-test'])
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')
    # 48 Articles, each article average 40~50 paragraph, each average 5~10 questions

    args = parser.parse_args()

    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion, "randomeSeed", args.randomSeed)

    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed is not None else None

    # if args.quantized:
        # tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
    # else:
        # tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     token=HF_TOKEN
        # )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def unique_path(path, i=0):  # 返回唯一路径
        if os.path.exists(path):  # 存在路径时，继续递归，生成新的路径
            # path = path.split("_")[:-1] if i != 0 else path
            return unique_path(path + "_" + str(i), i + 1)
        return path  # 不存在路径时，直接返回

    if os.path.exists(args.output):
        args.output = unique_path(args.output)  # 获取唯一输出路径

    kvcache_test(args)  # cag测试主程序
