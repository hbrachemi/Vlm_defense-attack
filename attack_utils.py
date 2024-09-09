classif_prompts = [
    "USER: <image> How would you label this image with a single descriptor?. ASSISTANT:",
    "USER: <image> image of  \nASSISTANT:",
    "USER: <image> If you were a classification model, which label would you attribute to the image? ASSISTANT:",
]

caption_prompts = [
    "USER: <image> Elaborate on the elements present in this image. ASSISTANT:",
    "USER: <image> Relate the main components of this picture in words. ASSISTANT:",
    "USER: <image> Offer a short description of the subjects present in this image. ASSISTANT:",
]

possible_prompts = classif_prompts+caption_prompts



class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.best_image = None

    def __call__(self, loss, image):
        
        if self.best_loss is None:
            self.best_loss = loss
            self.best_image = image
        
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.best_image = image
            self.counter = 0

