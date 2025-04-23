import gradio as gr
import torch
import transformers
import time
from safetensors.torch import save_file, load_file
import tempfile
from io import BytesIO
import logging

# Load the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


# Update the optimization function in demo.py to align with the notebook


def optimize_simon_says_prompt(
    input_text: str,
    number_of_simon_says_tokens: int,
    n_steps: int,
    lr: float,
    progress=gr.Progress(track_tqdm=False),  # Gradio progress tracking
) -> tuple[str, torch.Tensor]:
    """
    Optimize a Simon Says prompt based on the input text and display the optimization process.

    Parameters:
        input_text (str): The input text provided by the user.
        number_of_simon_says_tokens (int): Number of Simon Says tokens to optimize.
        n_steps (int): Number of optimization steps.
        lr (float): Learning rate for the optimization process.
        progress (gr.Progress): Gradio progress tracking.

    Returns:
        The optimized Simon Says prompt
    """
    # Tokenize the input text
    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        add_special_tokens=True,
    )
    embeddings = model.transformer.wte(tokens["input_ids"]).detach()

    # Initialize a random Simon Says prompt
    simon_says_prompt = torch.randn(
        1, number_of_simon_says_tokens, model.config.n_embd, requires_grad=True
    )
    optimizer = torch.optim.Adam([simon_says_prompt], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss: float = float("inf")
    best_simon_says_prompt: torch.Tensor = None

    progress(0, desc="Starting optimization...")
    time.sleep(1)

    for step in range(n_steps):
        optimizer.zero_grad()
        expanded_prompt = torch.cat([simon_says_prompt, embeddings], dim=1)
        logits = model(inputs_embeds=expanded_prompt).logits
        probs = torch.softmax(logits[:, simon_says_prompt.size(-2) - 1 : -1], dim=-1)
        ranks = (
            torch.sum(
                probs > probs.gather(2, tokens["input_ids"].unsqueeze(-1)), dim=-1
            )
            + 1
        )
        loss = loss_fn(
            logits[:, simon_says_prompt.size(-2) - 1 : -1].reshape(-1, logits.size(-1)),
            tokens["input_ids"].reshape(-1),
        )
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

        # If all ranks are 1, stop the optimization (perfect prediction)
        if torch.all(ranks == 1):
            break

    return best_simon_says_prompt


# Modify the download_tensor function to save the tensor as a safetensors file


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


def greedy_decode_with_ss_prompt(
    ss_prompt: torch.Tensor, progress=gr.Progress()
) -> str:
    """
    Perform greedy decoding using an uploaded optimized tensor and input text.

    Parameters:
        ss_prompt (torch.Tensor): The uploaded optimized tensor.
        progress (gr.Progress): Gradio progress tracking.

    Returns:
        str: The generated text.
    """
    generated_tokens = []
    all_logits = []

    progress(0, desc="Starting greedy decoding...")

    with torch.no_grad():
        for i in progress.tqdm(range(150), desc="Decoding..."):
            if len(generated_tokens) == 0:
                expanded_prompt = ss_prompt
            else:
                expanded_prompt = torch.cat(
                    [
                        ss_prompt,
                        model.transformer.wte(
                            torch.tensor(generated_tokens).unsqueeze(0)
                        ).detach(),
                    ],
                    dim=1,
                )

            logits = model(inputs_embeds=expanded_prompt).logits
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
) -> tuple[str, str]:
    """
    Optimize the Simon Says prompt, display the optimization process, and generate text based on the input text.

    Parameters:
        input_text (str): The input text provided by the user.
        number_of_simon_says_tokens (int): Number of Simon Says tokens to optimize.
        n_steps (int): Number of optimization steps.
        lr (float): Learning rate for the optimization process.

    Returns:
        tuple: The optimized Simon Says prompt and the greedy-decoded text.
    """
    optimized_prompt = optimize_simon_says_prompt(
        input_text=input_text,
        number_of_simon_says_tokens=number_of_simon_says_tokens,
        n_steps=n_steps,
        lr=lr,
    )

    # Generate text using the optimized prompt
    generated_text: str = greedy_decode_with_ss_prompt(optimized_prompt)

    return (
        generated_text,
        download_tensor(optimized_prompt),
    )  # Return the optimized tensor for download


def process_with_uploaded_tensor(
    input_text: str, uploaded_tensor: torch.Tensor
) -> tuple[str, str]:
    """
    Process the uploaded tensor and generate text based on the input text.

    Parameters:
        input_text (str): The input text provided by the user.
        uploaded_tensor (torch.Tensor): The uploaded optimized tensor.

    Returns:
        tuple: The generated text and the file path of the uploaded tensor.
    """
    generated_text = greedy_decode_with_ss_prompt(uploaded_tensor)
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

# Update the Gradio interface to include configurable parameters
demo = gr.Interface(
    theme=theme,
    title="Simon Says Prompt Optimization and Text Generation",
    fn=lambda input_text, number_of_simon_says_tokens, n_steps, lr, uploaded_file: (
        process_with_uploaded_tensor(input_text, upload_tensor(uploaded_file))
        if uploaded_file
        else process_and_generate(
            input_text, number_of_simon_says_tokens, n_steps, lr
        )
    ),
    inputs=[
        gr.Textbox(
            lines=5,
            placeholder="Enter your text here...",
            label="Input Text",
            value="Hello world! I'm Aldan, happy to be here.",
        ),
        gr.Slider(
            minimum=1, maximum=10, step=1, value=4, label="Number of Simon Says Tokens"
        ),
        gr.Slider(
            minimum=100,
            maximum=10000,
            step=100,
            value=5000,
            label="Number of Optimization Steps",
        ),
        gr.Slider(
            minimum=1e-5, maximum=1e-1, step=1e-5, value=1e-2, label="Learning Rate"
        ),
        gr.File(label="Upload Optimized Tensor (Optional)", type="binary"),
    ],
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.File(label="Download Optimized Tensor", type="filepath"),
    ],
    description="This demo optimizes a Simon Says prompt based on your input text, displays the optimization process, and generates text using the optimized prompt. Optionally, you can upload a pre-optimized tensor for inference.",
)

# Ensure the Gradio interface is correctly launched
if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
