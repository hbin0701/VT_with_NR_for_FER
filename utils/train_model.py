import torch
import sys
import os
from tqdm import tqdm

def train_model(model, log_file, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100):

    f = open(log_file, "w")

    model_path = os.path.split(log_file)[0]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    
    best_loss = float('inf')
    acc = []

    for epoch in range(1, num_epochs + 1):

        total_train_loss = 0.0
        total_test_loss = 0.0

        total_train_acc = 0
        total_test_acc = 0

        
        # Train
        model.train()

        scheduler.step()

        with tqdm(train_loader, unit="batch", file=sys.stdout) as t_epoch:
            for batch in t_epoch:
                t_epoch.set_description(f"Training at Epoch {epoch}")
                img, label = batch

                img = img.to(dev)
                label = label.to(dev)

                optimizer.zero_grad()

                output = model(img)
                loss = criterion(output, label)

                pred = torch.max(output, dim=1)[1]
                total_train_acc += (pred == label).to(torch.float).mean()

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                t_epoch.set_postfix(loss=loss.item(), accuracy= 100*float((pred == label).to(torch.float).mean()))
                
        average_train_loss = total_train_loss / len(train_loader)
        average_train_acc = total_train_acc / len(train_loader)

        print('\nEpoch {}: Avg. Train Loss: {:.4f}'.format(epoch, average_train_loss))
        print('Epoch {}: Avg. Train Acc.: {:.4f}'.format(epoch, average_train_acc))
        
        
        # Test
        model.eval()

        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as v_epoch:
                for batch in v_epoch:
                    v_epoch.set_description(f"Testing at Epoch {epoch}")
                    img, label = batch

                    img = img.to(dev)
                    label = label.to(dev)

                    output = model(img)
                    loss = criterion(output, label)

                    pred = torch.max(output, dim=1)[1]
                    total_test_acc += (pred == label).to(torch.float).mean()

                    total_test_loss += loss.item()
                    v_epoch.set_postfix(loss=loss.item(), accuracy= 100*float((pred == label).to(torch.float).mean()))

            average_test_loss = total_test_loss / len(test_loader)
            average_test_acc = total_test_acc / len(test_loader)

        acc.append(average_test_acc)
        print('\nEpoch {}: Avg. Test Loss: {:.4f}'.format(epoch, average_test_loss))
        print('Epoch {}: Avg. Test Acc.: {:.4f}'.format(epoch, average_test_acc))

        ## Saving
        ckpt = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': average_train_loss,
                'test_loss': average_test_loss,
                }

        torch.save(ckpt, os.path.join(model_path, f"last.pth"))

        if average_test_loss < best_loss:
            best_loss = average_test_loss
            torch.save(ckpt, os.path.join(model_path, f"best.pth"))

        f.write('Epoch {}: Avg. Train Loss: {:.4f}\n'.format(epoch, average_train_loss))
        f.write('Epoch {}: Avg. Train Acc.: {:.4f}\n'.format(epoch, average_train_acc))
        f.write('Epoch {}: Avg. Test Loss: {:.4f}\n'.format(epoch, average_test_loss))
        f.write('Epoch {}: Avg. Test Acc.: {:.4f}\n\n'.format(epoch, average_test_acc))
        f.flush()
        
    best_acc = max(acc).item()
    best_epoch = str(acc.index(best_acc))
    f.write('\n Best Test Accuracy at Epoch {}, {:.4f}'.format(best_epoch, best_acc))
    f.close()
