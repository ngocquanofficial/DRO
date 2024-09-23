import torch


def log_det(y_true, pred_list, num_models):
    mask_non_y_true = ~y_true.bool()  # Mask out the true class
    print(mask_non_y_true)
    log_dets = []
    
    for batch in range(pred_list[0].size(0)):  # Iterate over each batch
        masked_preds = []
        for i in range(num_models):
            mask_pred = pred_list[i][batch][mask_non_y_true[batch]]
            masked_preds.append(mask_pred)


        masked_preds = torch.stack(masked_preds)
        print(masked_preds)
        norm_preds = masked_preds / torch.norm(masked_preds, dim=1, keepdim=True)
        print(norm_preds)

        matrix = torch.matmul(norm_preds, norm_preds.t())
        log_det_val = torch.logdet(matrix + 1e-7 * torch.eye(matrix.shape[0]).detach() )
        print(1e-6 * torch.eye(matrix.shape[0]).detach() )
        print("DET")
        print(torch.logdet(matrix))
        print(log_det_val)
        log_dets.append(log_det_val)
        

    return torch.stack(log_dets).mean()

y_true = torch.tensor([[0, 0, 1]])
pred_list = torch.tensor([ [[1.0,2.0,3.0]], [[4.0,5.0,6.0]], [[7.0,8.0,9.0]] ])

print(log_det(y_true, pred_list, 3))
