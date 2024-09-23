import torch
import torch.nn.functional as F
import statistics

def log_det1(y_true, pred):
    mask_non_y_true = ~y_true.bool()  # Mask out the true class

    for batch in range(pred[0].size(0)):  # Iterate over each batch
        masked_preds = []
        for i in range(num_models):
            mask_pred = pred_list[i][batch][mask_non_y_true[batch]]
            masked_preds.append(mask_pred)


        masked_preds = torch.stack(masked_preds)
        print(masked_preds)
        norm_preds = masked_preds / torch.norm(masked_preds, dim=1, keepdim=True)
        print(norm_preds)
        pred = norm_preds
        # Calculate pairwise cosine similarity
        for i in range(n):
            for j in range(i + 1, n):
                print("i,j", pred[i], pred[j])
                cos_sim = F.cosine_similarity(pred[i].unsqueeze(0), pred[j].unsqueeze(0))
                total_similarity += cos_sim.item()  # Extract the scalar value
                count += 1
                print(cos_sim.item())
        
        # Calculate the average cosine similarity
        average_cosine_similarity = total_similarity / count if count > 0 else 0
        return average_cosine_similarity
        

    # return torch.stack(log_dets).mean()

def log_det(y_true, pred, num_models):
    mask_non_y_true = ~y_true.bool()  # Mask out the true class
    log_dets = []
    
    for batch in range( pred[0].shape[0] ):  # Iterate over each batch
        masked_preds = []
        for i in range(num_models):
            mask_pred = pred[i][batch][mask_non_y_true[batch]]
            masked_preds.append(mask_pred)

        masked_preds = torch.stack(masked_preds)
        print(masked_preds)
        norm_preds = masked_preds / torch.norm(masked_preds, dim=1, keepdim=True)
        print(norm_preds)
        matrix = torch.matmul(norm_preds, norm_preds.t())
        log_det_val = torch.logdet(matrix + 1e-7 * torch.eye(num_models).to(pred[0].device))
        log_dets.append(log_det_val)

    return torch.stack(log_dets).mean()


def cal_cosine_similarity(y_true, pred, num_models):
    min_cosines = []
    max_cosines = []
    avg_cosines = []

    mask_non_y_true = ~y_true.bool()  # Mask out the true class

    
    for batch in range( pred[0].shape[0] ):  # Iterate over each batch
        masked_preds = []
        for i in range(num_models):
            mask_pred = pred[i][batch][mask_non_y_true[batch]]
            masked_preds.append(mask_pred)

        masked_preds = torch.stack(masked_preds)
        print(masked_preds)
        norm_preds = masked_preds / torch.norm(masked_preds, dim=1, keepdim=True)
        print(norm_preds)
        nonmaximal = norm_preds

        n = nonmaximal.shape[0]
        total_similarity = 0
        count = 0
        max_cosine = 0
        min_cosine = 1

        
        # Calculate pairwise cosine similarity
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = F.cosine_similarity(nonmaximal[i].unsqueeze(0), nonmaximal[j].unsqueeze(0))
                current = cos_sim.item()  # Extract the scalar value
                total_similarity += current

                if current > max_cosine :
                    max_cosine = current
                if current < min_cosine :
                    min_cosine = current

                count += 1
        
        # Calculate the average cosine similarity
        average_cosine_similarity = total_similarity / count if count > 0 else 0
        avg_cosines.append(average_cosine_similarity)
        min_cosines.append(min_cosine)
        max_cosines.append(max_cosine)
        print(average_cosine_similarity, max_cosine, min_cosine)
    
    return sum(avg_cosines)/len(avg_cosines), statistics.median(max_cosines), statistics.median(min_cosines)


y_true = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]])
pred1 = torch.tensor([[1.0,1.0,1.0,1.0], [4.0, 1.0, 0.0, 0.0]])
pred2 = torch.tensor([[1.0,1.0,0.0,0.0], [4.0, 1.0, 0.0, 0.0]])
pred3 = torch.tensor([[0.0,0.0,0.0,1.0], [4.0, 0.0, 0.0, 1.0]])

pred_list = [pred1, pred2, pred3]

print(cal_cosine_similarity(y_true, pred_list, 3))

# print(torch.nn.functional.softmax(pred3, dim= -1))
