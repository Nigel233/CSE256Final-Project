import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc1, best_acc2 = 0, 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target1, target2 = batch.text, batch.label1, batch.label2
            with torch.no_grad():
                feature.t_()
                target1.sub_(1)
                target2.sub_(1)

            if args.cuda:
                feature, target1, target2 = feature.cuda(), target1.cuda(), target2.cuda()

            optimizer.zero_grad()
            logit1, logit2 = model(feature)
            loss = F.cross_entropy(logit1, target1) + F.cross_entropy(logit2, target2)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects1 = (torch.max(logit1, 1)[1].view(target1.size()).data == target1.data).sum()
                accuracy1 = 100.0 * corrects1/batch.batch_size

                corrects2 = (torch.max(logit2, 1)[1].view(target2.size()).data == target2.data).sum()
                accuracy2 = 100.0 * corrects2/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc1: {:.4f}%({}/{}) acc2: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy1,
                                                                             corrects1,
                                                                             batch.batch_size,
                                                                             accuracy2,
                                                                             corrects2,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc1, dev_acc2 = eval(dev_iter, model, args)
                if dev_acc1 > best_acc1:
                    best_acc1 = dev_acc1
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                elif dev_acc2 > best_acc2:
                    best_acc2 = dev_acc2
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects1, avg_loss1 = 0, 0
    corrects2, avg_loss2 = 0, 0
    for batch in data_iter:
        feature, target1, target2 = batch.text, batch.label1, batch.label2
        with torch.no_grad():
            feature.t_()
            target1.sub_(1)
            target2.sub_(1)
        if args.cuda:
            feature, target1, target2 = feature.cuda(), target1.cuda(), target2.cuda()

        logit1, logit2 = model(feature)
        loss1 = F.cross_entropy(logit1, target1, size_average=False)
        loss2 = F.cross_entropy(logit2, target2, size_average=False)
        
        avg_loss1 += loss1.item()
        avg_loss2 += loss2.item()
        corrects1 += (torch.max(logit1, 1)
                     [1].view(target1.size()).data == target1.data).sum()
        corrects2 += (torch.max(logit2, 1)
                     [1].view(target2.size()).data == target2.data).sum()

    size = len(data_iter.dataset)
    avg_loss1 /= size
    avg_loss2 /= size
    accuracy1 = 100.0 * corrects1/size
    accuracy2 = 100.0 * corrects2/size
    print('\nEvaluation - loss: {:.6f}  acc1: {:.4f}%({}/{}) acc2: {:.4f}%({}/{})\n'.format(avg_loss1, 
                                                                       accuracy1, 
                                                                       corrects1, 
                                                                       size, 
                                                                       accuracy2, 
                                                                       corrects2, 
                                                                       size))
    return accuracy1, accuracy2


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
