import torch
from torch.autograd import Variable
from utils.strategy import step_lr, accuracy
import time

def train(model,arr_train_loader,arr_test_loader, criterion, args):
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    max_val_acc = 0
    train_draw_acc = []
    val_draw_acc = []
    lr = args.lr

    for epoch in range(args.max_epochs):

        ep_start = time.time()
        lr = step_lr(epoch, lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        model.train()
        top1_sum = 0
        for i, (signal, label) in enumerate(arr_train_loader):
            input = Variable(signal).cuda()
            target = Variable(label).cuda().long()

            output = model(input)            # inference
            
            loss = criterion(output, target) # 计算交叉熵损失
            optimizer.zero_grad()
            loss.backward()                  # 反传
            optimizer.step()

            top1 = accuracy(output.data, target.data, topk=(1,))  # 计算top1分类准确率
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]

            print('Epoch [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                    %(epoch+1, args.max_epochs, lr, train_loss_sum/sum, train_top1_sum/sum))

            sum = 0
            train_loss_sum = 0
            train_top1_sum = 0

        # train_draw_acc.append((top1_sum/len(arr_train_loader)).cpu().numpy())
        
        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < args.max_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1 = eval(model, arr_test_loader, criterion)
            val_draw_acc.append(val_top1.cpu().numpy())
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f'
                    %(epoch+1, args.max_epochs, val_loss, val_top1, val_time*60, max_val_acc))
            print('epoch time: {}s'.format(epoch_time*60))
            if val_top1[0].data > max_val_acc:
                max_val_acc = val_top1[0].data
                print('Taking snapshot...')
                torch.save(model.state_dict(), '{}/{}/semi_{}_{}_best.pth'.format('checkpoints', args.method,args.dataset, args.rate))
              
    return max_val_acc.item()
    
# validation 测试部分
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1