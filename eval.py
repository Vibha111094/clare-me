import torch

from dataset import Dataset

def evaluate_model(model, test_data):
    
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    outlier_list = []
    y_pred = []

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        id = 0
        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              
              output = model(input_id, mask)
              if (output.argmax(dim=1)!=test_label):
                outlier_list.append(id)
              y_pred.append(output.argmax(dim=1))
                         

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              id = id+1
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return outlier_list,y_pred