import time
import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = torch.nn.CrossEntropyLoss()

# if model already trained, then just load it:
if os.path.exists('human_dog_distinctor_new.pt'):
    print('Shall load pretrained model in next cell.')


else:
    torch.cuda.empty_cache()
    # specify optimizer (stochastic gradient descent) and learning rate = 0.001
    optimizer = optim.Adam(hfdetector.parameters(), lr=0.00003)

    start = time.time()
    print(f'Training started at {time.ctime()}')

    # number of epochs to train the model
    n_epochs = 50
    stop_criterion = 5
    valid_loss_min = np.Inf
    early_stop_count = 0

    if torch.cuda.device_count() >= 2:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        hfdetector = torch.nn.DataParallel(hfdetector)
    elif torch.cuda.is_available():
        hfdetector = hfdetector.cuda()

    # Time meter
    batch_time = AverageMeter()
    data_time = AverageMeter()

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # early stop mechanism:
        if early_stop_count >= stop_criterion:
            print(
                f'Validation loss stops decresing for {stop_criterion} epochs, early stop triggered.')
            break

        ###################
        # train the model #
        ###################
        hfdetector.train()
        e = time.time()
        for data, target in train_loader:

            # measure data loading time
            data_time.update(time.time() - e)

            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = hfdetector(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

            # measure elapsed time
            batch_time.update(time.time() - e)
            e = time.time()

        ######################
        # validate the model #
        ######################
        hfdetector.eval()

        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():
                data, target = data.cuda(
                    non_blocking=True), target.cuda(non_blocking=True)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = hfdetector(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(hfdetector.state_dict(), 'human_dog_distinctor_new.pt')
            valid_loss_min = valid_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        # print time
        print(
            f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) Data {data_time.val:.3f}s ({data_time.avg:.3f}s)')

    end = time.time()
    t = int(end - start)
    print(
        f'Training ended at {time.ctime()}, total training time is {t//3600}hours {(t%3600)//60}minutes {(t%3600)%60} seconds.')
