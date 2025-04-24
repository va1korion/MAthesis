


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})

    messages = [{"role": "user", "content": "Для чего нужна токенизация?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True))
    response = llm.invoke(messages)
    return {"answer": response.content}
