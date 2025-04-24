import gradio as gr
import torch
import transformers
import time
from safetensors.torch import save_file, load_file
import tempfile
from io import BytesIO
import logging

# Add a dropdown to select the model
model_options = [
    "openai-community/gpt2",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-14m",
]


def load_model_and_tokenizer(model_name):
    """
    Load the tokenizer and model based on the selected model name.

    Parameters:
        model_name (str): The name of the model to load.

    Returns:
        tuple: The loaded tokenizer and model.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def get_embeddings(
    input_ids: torch.Tensor, model: transformers.PreTrainedModel
) -> torch.Tensor:
    """
    Get the embeddings for the input IDs.

    Parameters:
        input_ids (torch.Tensor): The input IDs for which to get the embeddings.
        model (transformers.PreTrainedModel): The model to use for generating embeddings.

    Returns:
        torch.Tensor: The embeddings for the input IDs.
    """
    return model.get_input_embeddings()(input_ids).detach()


def optimize_simon_says_prompt(
    input_text: str,
    number_of_simon_says_tokens: int,
    n_steps: int,
    lr: float,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    add_eos_token: bool,
    progress=gr.Progress(track_tqdm=False),  # Gradio progress tracking
) -> tuple[str, torch.Tensor]:
    """
    Optimize a Simon Says prompt based on the input text and display the optimization process.

    Parameters:
        input_text (str): The input text provided by the user.
        number_of_simon_says_tokens (int): Number of Simon Says tokens to optimize.
        n_steps (int): Number of optimization steps.
        lr (float): Learning rate for the optimization process.
        model (transformers.PreTrainedModel): The model to use for optimization.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        add_eos_token (bool): Whether to add an EOS token to the input text.
        progress (gr.Progress): Gradio progress tracking.

    Returns:
        The optimized Simon Says prompt
    """
    torch.manual_seed(42)  # Set a random seed for reproducibility

    # Check if the EOS token checkbox is selected
    if add_eos_token:
        # We could've also used the tokenizer.eos_token_id, but this is easier because we don't need to potentially handle padding attention masks, batching issues, etc.
        input_text += tokenizer.eos_token

    # Tokenize the input text
    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        add_special_tokens=False,
    )
    embeddings = get_embeddings(tokens["input_ids"], model)

    # Initialize a random Simon Says prompt
    simon_says_prompt = torch.randn(
        1, number_of_simon_says_tokens, embeddings.size(-1), requires_grad=True
    )

    dummy_prompt = torch.zeros_like(
        simon_says_prompt[..., 0:1, :], requires_grad=False
    )  # Add an extra dimension

    attention_mask = torch.ones_like(
        torch.cat([dummy_prompt, simon_says_prompt, embeddings], dim=1)[:, :, 0],
        device=simon_says_prompt.device,
        requires_grad=False,
    )
    # Set the first token to 0 in the attention mask
    attention_mask[:, 0] = 0

    optimizer = torch.optim.Adam([simon_says_prompt], lr=lr)

    best_loss: float = float("inf")
    best_simon_says_prompt: torch.Tensor = None

    progress(0, desc="Starting optimization...")
    time.sleep(1)

    for step in range(n_steps):
        optimizer.zero_grad()
        expanded_prompt = torch.cat(
            [dummy_prompt, simon_says_prompt, embeddings], dim=1
        )
        logits = model(
            inputs_embeds=expanded_prompt, attention_mask=attention_mask
        ).logits
        probs = torch.softmax(logits[:, -embeddings.size(-2) - 1 : -1], dim=-1)
        ranks = (
            torch.sum(
                probs > probs.gather(2, tokens["input_ids"].unsqueeze(-1)), dim=-1
            )
            + 1
        )

        # If all ranks are 1, stop the optimization (perfect prediction)
        if torch.all(ranks == 1):
            best_simon_says_prompt = simon_says_prompt.detach().clone()
            break

        loss = torch.functional.F.cross_entropy(
            input=logits[:, -embeddings.size(-2) - 1 : -1].reshape(-1, logits.size(-1)),
            target=tokens["input_ids"].reshape(-1),
            reduction="none",
        )
        # Multiply the loss by the ranks to give more weight to the tokens with higher ranks - this is to speed up the optimization process and avoid getting stuck in local minima
        # Weights should be between 0 and 1 - we can normalize the ranks to get weights and then apply softmax to get the final weights as a more stable distribution
        token_weights = ranks.float() / ranks.float().max()
        print(f"Token Ranks: {ranks}")
        print(f"Token Weights: {token_weights}")
        loss = loss * token_weights.reshape(-1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        avg_rank = ranks.float().mean().item()
        progress(
            step / n_steps,
            desc=f"Step {step}, Loss: {loss.item():.4f}, Avg Rank: {avg_rank:.2f}, Max Rank: {ranks.max().item()}",
        )

        logging.info(
            f"Step {step}, Loss: {loss.item():.4f}, Avg Rank: {avg_rank:.2f}, Max Rank: {ranks.max().item()}"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_simon_says_prompt = simon_says_prompt.detach().clone()

    else:
        # Show a Gradio warning saying that the optimization did not converge
        gr.Warning(
            "The optimization did not converge. The prompt will not generate the expected output."
        )

    return best_simon_says_prompt


def download_tensor(tensor):
    """
    Save a tensor to a safetensors file for download.

    Parameters:
        tensor (torch.Tensor): The tensor to be saved.

    Returns:
        str: The file path of the saved tensor.
    """
    file_path = "optimized_tensor.safetensors"
    save_file({"optimized_tensor": tensor}, file_path)
    return file_path


def upload_tensor(file):
    """
    Load a tensor from an uploaded safetensors file.

    Parameters:
        file (bytes): The uploaded file containing the safetensors data.

    Returns:
        torch.Tensor: The loaded tensor.

    Raises:
        ValueError: If the safetensors file is invalid or the header is too large.
    """
    if isinstance(file, bytes):
        file = BytesIO(file)  # Wrap bytes in a BytesIO object

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(
            file.read()
        )  # Directly write the BytesIO content to the temporary file
        temp_file.flush()

        try:
            tensor_data = load_file(temp_file.name)
        except Exception as e:
            raise ValueError(f"Failed to load safetensors file: {e}")

        if "optimized_tensor" not in tensor_data:
            raise ValueError(
                "The safetensors file does not contain the expected 'optimized_tensor' key."
            )

        return tensor_data["optimized_tensor"]


@torch.inference_mode()
def greedy_decode_with_simon_says_prompt(
    simon_says_prompt: torch.Tensor,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    progress=gr.Progress(),
) -> str:
    """
    Perform greedy decoding using an uploaded optimized tensor and input text.

    Parameters:
        simon_says_prompt (torch.Tensor): The uploaded optimized tensor.
        model (transformers.PreTrainedModel): The model to use for decoding.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for decoding.
        progress (gr.Progress): Gradio progress tracking.

    Returns:
        str: The generated text.
    """
    generated_tokens = []
    all_logits = []

    progress(0, desc="Starting greedy decoding...")

    # Add an extra dimension with all 0s to the start of the prompt - this is just a bugfix because GPT-2 can't handle a prompt of size 1 (still investigating why)
    dummy_prompt = torch.zeros_like(
        simon_says_prompt[..., 0:1, :]
    )  # Add an extra dimension
    simon_says_prompt_with_dummy = torch.cat(
        [
            dummy_prompt,
            simon_says_prompt,
        ],
        dim=1,
    )

    for i in progress.tqdm(range(100), desc="Decoding..."):
        if len(generated_tokens) == 0:
            expanded_prompt = simon_says_prompt_with_dummy
        else:
            expanded_prompt = torch.cat(
                [
                    simon_says_prompt_with_dummy,
                    get_embeddings(
                        torch.tensor(
                            generated_tokens, device=simon_says_prompt.device
                        ).unsqueeze(0),
                        model,
                    ),
                ],
                dim=1,
            )

        attention_mask = torch.ones_like(
            expanded_prompt[:, :, 0], device=simon_says_prompt.device
        )
        # Set the first token to 0 in the attention mask
        attention_mask[:, 0] = 0

        logits = model(
            inputs_embeds=expanded_prompt,
            attention_mask=attention_mask,
        ).logits
        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax().item()

        logging.info(
            f"Step {i}, Next Token: {next_token}, Logit: {next_token_logits[next_token].item()}"
        )

        generated_tokens.append(next_token)
        all_logits.append(next_token_logits)

        if next_token == tokenizer.eos_token_id:
            break

    generated_tokens = torch.tensor(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def process_and_generate(
    input_text: str,
    number_of_simon_says_tokens: int,
    n_steps: int,
    lr: float,
    model_name: str,
    add_eos_token: bool,
) -> tuple[str, str]:
    """
    Optimize the Simon Says prompt, display the optimization process, and generate text based on the input text.

    Parameters:
        input_text (str): The input text provided by the user.
        number_of_simon_says_tokens (int): Number of Simon Says tokens to optimize.
        n_steps (int): Number of optimization steps.
        lr (float): Learning rate for the optimization process.
        model_name (str): The name of the model to load.
        add_eos_token (bool): Whether to add an EOS token to the input text.

    Returns:
        tuple: The optimized Simon Says prompt and the greedy-decoded text.
    """
    tokenizer, model = load_model_and_tokenizer(model_name)
    optimized_prompt = optimize_simon_says_prompt(
        input_text=input_text,
        number_of_simon_says_tokens=number_of_simon_says_tokens,
        n_steps=n_steps,
        lr=lr,
        model=model,
        tokenizer=tokenizer,
        add_eos_token=add_eos_token,
    )

    # Generate text using the optimized prompt
    generated_text: str = greedy_decode_with_simon_says_prompt(
        optimized_prompt, model, tokenizer
    )

    return (
        generated_text,
        download_tensor(optimized_prompt),
    )  # Return the optimized tensor for download


def process_with_uploaded_tensor(
    input_text: str, uploaded_tensor: torch.Tensor, model_name: str
) -> tuple[str, str]:
    """
    Process the uploaded tensor and generate text based on the input text.

    Parameters:
        input_text (str): The input text provided by the user.
        uploaded_tensor (torch.Tensor): The uploaded optimized tensor.
        model_name (str): The name of the model to load.

    Returns:
        tuple: The generated text and the file path of the uploaded tensor.
    """
    tokenizer, model = load_model_and_tokenizer(model_name)
    generated_text = greedy_decode_with_simon_says_prompt(
        uploaded_tensor, model, tokenizer
    )
    return generated_text, None


theme = gr.themes.Soft(
    primary_hue="fuchsia",
    secondary_hue="cyan",
    neutral_hue="gray",
    radius_size="none",
    font=[
        gr.themes.GoogleFont("IBM Plex Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("IBM Plex Mono"),
        "ui-monospace",
        "Consolas",
        "monospace",
    ],
)

# Update the Gradio interface to include the model selection dropdown
demo = gr.Interface(
    theme=theme,
    title="Simon Says Prompt Optimization and Text Generation",
    fn=lambda input_text, model_name, number_of_simon_says_tokens, n_steps, lr, add_eos_token, uploaded_file: (
        process_with_uploaded_tensor(
            input_text, upload_tensor(uploaded_file), model_name
        )
        if uploaded_file
        else process_and_generate(
            input_text,
            number_of_simon_says_tokens,
            n_steps,
            lr,
            model_name,
            add_eos_token,
        )
    ),
    inputs=[
        gr.Textbox(
            lines=5,
            placeholder="Enter your text here...",
            label="Input Text",
            value="Hello world! I'm Aldan, happy to be here.",
            info="Provide the text for which you want to optimize the Simon Says prompt. This text will be used as the target for generating the Simon Says Prompt.",
        ),
        gr.Dropdown(
            choices=model_options,
            value="EleutherAI/pythia-160m",
            label="Select Model",
            interactive=True,
            info="Choose a pre-trained language model to use for optimization and text generation. Each model has different capabilities and sizes.",
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=4,
            label="Number of Simon Says Prompt Tokens",
            info="Specify the number of tokens to include in the Simon Says prompt. Bigger sizes may make it easier to optimize, but they take up more space.",
        ),
        gr.Slider(
            minimum=100,
            maximum=10000,
            step=100,
            value=10000,
            label="Patience",
            info="Set the maximum number of steps for the optimization process. It will stop early if the optimization converges before reaching this number, but if it reaches the limit, it will stop without converging.",
        ),
        gr.Slider(
            minimum=1e-5,
            maximum=1e-1,
            step=1e-5,
            value=1e-1,
            label="Learning Rate",
            info="Adjust the learning rate for the optimization algorithm. This controls how quickly the optimization converges but can also lead to instability if set too high.",
        ),
        gr.Checkbox(
            label="Add EOS Token",
            value=False,
            interactive=True,
            info="Enable this option to append an End-Of-Sequence (EOS) token to the input text. This can help models better understand the input context.",
        ),
        gr.File(
            label="Upload Optimized SS Prompt (Optional)",
            type="binary",
            file_count="single",
            file_types=[".safetensors"],
        ),
    ],
    outputs=[
        gr.Textbox(
            label="Generated Text",
            info="The text generated by the model using the optimized Simon Says prompt.",
        ),
        gr.File(
            label="Download Optimized SS Prompt",
            type="filepath",
        ),
    ],
    description="This application allows you to optimize a Simon Says prompt based on your input text using advanced machine learning techniques. You can visualize the optimization process and generate text using the optimized prompt. Additionally, you can upload a pre-optimized tensor for direct inference (if you do, the other parameters will be ignored).",
)

# Ensure the Gradio interface is correctly launched
if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
