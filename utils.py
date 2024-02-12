import torch

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location="cpu")
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def format_result(result):
    if result==0:
        return("Regular conversations")
    elif result==1:
        return("Product-related conversations")
    elif result==2:
        return("Subscription-related conversations")
    elif result==3:
        return("Suicide")
    elif result==4:
        return("Non-mental health topics")