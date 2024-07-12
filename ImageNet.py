.import pickle 
import os 
from torch.utils.data import Dataset
from PIL import Image


class ImageNet_dataset(Dataset):
    
    """ImageNet Dataset.
    Generates images and a list containing the corresponding image label and bounding box coordinates in the format [image_label,x_min,y_min,x_max,y_max]
    The image labels are converted into the class id, please refer to the ImageNet synset_mapping file
    Notes:
    * This dataset class assumes the image_ids and corresponding list are orginized in a dictionnary (image_id : list) and serialized in a pickle file for data loading time optimization
    * One image can have multiple lables and bounding boxes"""

    def __init__(self,data_dict_path,db_path='./'):
        """
        Args:
            data_dict_path (string): Path to the pickle file dictionnary containing image ids and corresponding labels and bounding box scores.
            db_path (string): Directory with all the images, default assumes db_path is the same directory where the code is executed.
        """
        
        self.db_path = db_path
        with open(data_dict_path,'rb') as file: 
        	data_binary = pickle.load(file) 
        	self.data = pickle.loads(data_binary)
        
    def __len__(self):
        return len(self.data_dict_path)

    def __getitem__(self, idx):
        
        image_name = list(self.data.keys())[idx]
        image_gt = self.data[list(self.data.keys())[idx]]
        img_name = os.path.join(self.db_path,image_name+'.JPEG')
        
        image = Image.open(img_name)
        n_labels = len(image_gt) // 5
        img_bounding_box = [[image_gt[1+5*i],image_gt[2+5*i],image_gt[3+5*i],image_gt[4+5*i]] for i in range(n_labels)]
        img_class = [image_gt[0+5*i] for i in range(n_labels)]        
        
        return image, img_class, img_bounding_box
