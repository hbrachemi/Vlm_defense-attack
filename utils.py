from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def wrap_text(draw, text, font, max_width):
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and draw.textbbox((0, 0), line + words[0], font=font)[2] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())
    return lines

def text_to_image(text, font_path='arial.ttf', font_size=20, image_size=(224, 224), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    img = Image.new('RGB', image_size, color=bg_color)

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    max_width = image_size[0] - 20
    lines = wrap_text(draw, text, font, max_width)
    total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)

    y = (image_size[1] - total_text_height) // 2
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x = (image_size[0] - line_width) // 2
        draw.text((x, y), line, fill=text_color, font=font)
        y += draw.textbbox((0, 0), line, font=font)[3]

    return img

def generate_embeddings_using_hooks(model,processor,inputs,target_layer,input_output='output'):

        print(f"extracting {input_output} from layer: {target_layer}")

        list_acti = []

        if input_output == 'input':
                activations = []
                def hook_fn(module, input, output):
                    activations.append(input)
        else:
                activations = []
                def hook_fn(module, input, output):
                    activations.append(output)

        hook_handle = target_layer.register_forward_hook(hook_fn)
        inputs = processor(**inputs ,return_tensors="pt")
        output = model(**inputs)
        hook_handle.remove()
        activations_np = [act[0].cpu().detach().numpy() for act in activations]
        list_acti.append(activations_np)
        return list_acti
