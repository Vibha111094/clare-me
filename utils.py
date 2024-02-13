import torch

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location="cpu")
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def format_result(result):
    result = result.item()
    mapping = {
        0: "Regular conversations",
        1: "Product-related conversations",
        2: "Subscription-related conversations",
        3: "Suicide",
        4: "Non-mental health topics"
    }
    return mapping.get(result)