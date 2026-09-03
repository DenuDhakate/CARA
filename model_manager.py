import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM


# Model names
SMALL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MEDIUM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


small_model = None
small_tokenizer = None

medium_model = None
medium_tokenizer = None


def clear_gpu():

    gc.collect()
    torch.cuda.empty_cache()


def load_small_model():

    global small_model
    global small_tokenizer
    global medium_model
    global medium_tokenizer

    # If already loaded, don't reload
    if small_model is not None:
        return

    # Remove medium model if loaded
    if medium_model is not None:

        del medium_model
        del medium_tokenizer

        medium_model = None
        medium_tokenizer = None

        clear_gpu()

    print("Loading Small Model...")

    small_tokenizer = AutoTokenizer.from_pretrained(
        SMALL_MODEL_NAME
    )

    small_model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    print("Small Model Ready!")


def load_medium_model():

    global medium_model
    global medium_tokenizer
    global small_model
    global small_tokenizer

    # If already loaded, don't reload
    if medium_model is not None:
        return

    # Remove small model if loaded
    if small_model is not None:

        del small_model
        del small_tokenizer

        small_model = None
        small_tokenizer = None

        clear_gpu()

    print("Loading Medium Model...")

    medium_tokenizer = AutoTokenizer.from_pretrained(
        MEDIUM_MODEL_NAME
    )

    medium_model = AutoModelForCausalLM.from_pretrained(
        MEDIUM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    print("Medium Model Ready!")


def generate_response(
    query,
    route,
    max_new_tokens=200
):

    if route == "SMALL":

        load_small_model()

        model = small_model
        tokenizer = small_tokenizer

    else:

        load_medium_model()

        model = medium_model
        tokenizer = medium_tokenizer


    messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response