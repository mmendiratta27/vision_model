from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import os, sys

model_id = "vikhyatk/moondream2"

revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Language model to interpret request
import dspy
ollama_phi3 = dspy.OllamaLocal(model='phi3')

dspy.settings.configure(lm=ollama_phi3)

class BasicQA(dspy.Signature):
    """Based on the request, determine what the user is looking for. """

    question = dspy.InputField()
    answer = dspy.OutputField(desc="1 to 5 words only")

class MatchingItems(dspy.Signature):
    """ Determine if we found the item based on a description of an image and the item we are looking for """

    question = dspy.InputField()
    answer = dspy.OutputField(desc="1 to 5 words only")

# Define the predictor.
generate_item = dspy.Predict(BasicQA)
find_item = dspy.ChainOfThought(MatchingItems)

question = "Can you help me find the blue backpack?"

# Call the predictor on the same input.
pred = generate_item(question=question)

des_item = pred.answer

print(des_item)

# naive representation of constant video input
images = ['/Users/mananmendiratta/Downloads/red.jpg', '/Users/mananmendiratta/Downloads/blue.jpg', '/Users/mananmendiratta/Downloads/green.jpg']

for i, item in enumerate(images):
    image = Image.open(item)
    enc_image = model.encode_image(image)
    print(f"IMAGE DESCRIPTION {i}\n")
    # start = time.time()
    ans = model.answer_question(enc_image, "Describe this image.", tokenizer)
    print(ans + '\n')

    foundQ = "Tell me if we found our desired item in one word: 'yes' or 'no'. Here is the desired item:" + des_item + "Here is the description: " + ans

    pred = find_item(question=foundQ)

    if 'Yes' or 'yes' in pred.answer:
        print("We found it!")
        # print(time.time() - start)

        # break feedback to user whenever desired item is found
        break
    else:
        print(pred.answer)
