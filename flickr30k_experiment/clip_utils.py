import torch
import open_clip

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# one function to load the CLIP model in the project
def load_clip_model(model_name='ViT-B-32', pretrained='openai', quick_gelu=False):
  """
  Returns:
    model - the frozen CLIP VisionTransformer on DEVICE
    preprocess - the torchvision-style preprocess transform
  """
  model, _, preprocess = open_clip.create_model_and_transforms(
      model_name,
      pretrained=pretrained,
      quick_gelu=quick_gelu
  )
  model = model.to(DEVICE)
  model.eval()
  for p in model.parameters():
      p.requires_grad = False
  return model, preprocess