model_loader.py # video/model_loader.py
# Think of this file as our AI Librarian. 📚
# Its only job is to find, download, and prepare the powerful, pre-trained
# AI models that we need for our video analysis.

import torch
from ultralytics import YOLO

class VideoModelLoader:
    """
    A helper class that fetches our AI models from the "cloud library"
    and gets them ready for our project.
    """
    def __init__(self, model_name="facebookresearch/pytorchvideo:main", model_variant="x3d_m"):
        """
        Initializes our librarian and notes which models we'll be checking out.
        """
        self.yolo_model_name = "yolov8n.pt"  # A small, fast model for spotting people.
        self.action_model_repo = model_name
        self.action_model_variant = model_variant
        self.device = self._get_device()
        print(f"All models will be loaded onto the '{self.device}' device.")

    def _get_device(self):
        """Checks if we have a powerful GPU to use, otherwise defaults to the CPU."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_yolo_model(self):
        """
        Checks out the YOLOv8 model, which is an expert at finding people in images.
        The 'ultralytics' library handles the download for us automatically.
        """
        print(f"Finding our person-spotting expert: '{self.yolo_model_name}'...")
        try:
            model = YOLO(self.yolo_model_name)
            print("✅ Person-spotter is ready!")
            return model
        except Exception as e:
            print(f"❌ Oh no, couldn't load the YOLO model. Error: {e}")
            return None

    def load_action_model(self):
        """
        Checks out the X3D model, an expert at recognizing human actions,
        and modifies it for our specific distress classes.
        """
        print(f"Finding our action-recognition expert: '{self.action_model_variant}'...")
        try:
            model = torch.hub.load(
                self.action_model_repo,
                model=self.action_model_variant,
                pretrained=True
            )
            
            # Replace the final classification layer for our specific classes
            num_classes = 5 # Corresponds to "Normal_activity", "Fighting", etc.
            model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, num_classes)

            # We tell the model to get ready for analysis (not training) and move it to our device.
            model = model.eval().to(self.device)
            print("✅ Action-recognizer is ready!")
            return model
        except Exception as e:
            print(f"❌ Oh no, couldn't load the action model from PyTorch Hub. Error: {e}")
            print("   Please check your internet connection and the model name.")
            return None

    def get_model(self):
        """Helper to get the action model directly."""
        return self.load_action_model()

    def get_device(self):
        """Helper to get the processing device."""
        return self.device

if __name__ == '__main__':
    # A quick test to make sure our librarian can find all the books.
    # You can run this file directly to check if the models download correctly.
    print("--- Testing the AI Librarian ---")
    loader = VideoModelLoader()
    yolo = loader.load_yolo_model()
    action_model = loader.load_action_model()

    if yolo and action_model:
        print("\n--- All models checked out successfully! ---")
    else:
        print("\n--- There was a problem checking out the models. ---")