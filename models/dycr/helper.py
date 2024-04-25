from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy


def base_train(session, model, trainloader, optimizer, scheduler, epoch, args, trip_loss, dim=4, exemplar=None, save_sample=10):
    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        output, logits = model(data)

        combine = logits.detach().cpu().numpy()
        combine = numpy.reshape(combine, (-1, dim, int(model.module.num_features/dim)))

        items = []
        answer = []

        for item in range(combine.shape[0]):
            # Decompose into category map (R) and context map (Q).
            Q, R = numpy.linalg.qr(combine[item], mode="complete")

            # Reshape context map back to feature shape.
            skeleton = torch.from_numpy(R.reshape(model.module.num_features)).float()
            items.append(skeleton.view(1, model.module.num_features).requires_grad_().cuda())

            # Obtain corresponding class prototype.
            answer.append(model.module.fc.weight.data[train_label[item]].view(1, model.module.num_features))

        proto_list = torch.cat(items, dim=0)
        answer = torch.cat(answer, dim=0)

        # Calculate L2 distance between each sample's category information with class prototypes
        dist_map = torch.cdist(logits.view(logits.shape[0], -1), model.module.fc.weight.data.view(model.module.fc.weight.data.shape[0], -1), p=2).cuda()
        tri_loss = torch.tensor([0.], requires_grad=True).cuda()

        for i in range(logits.shape[0]):
            # Obtain the first two most similar prototypes for each sample
            values, indices = torch.topk(dist_map[i], 2, largest=False)
            ground_truth = train_label[i].item()

            if indices[0].item() == ground_truth:
                tri_loss = tri_loss + trip_loss(proto_list[i].unsqueeze(0), answer[i].unsqueeze(0)
                                                        , model.module.fc.weight.data[indices[1].item()].unsqueeze(0))
            else:
                tri_loss = tri_loss + trip_loss(proto_list[i].unsqueeze(0), answer[i].unsqueeze(0)
                                                        , model.module.fc.weight.data[indices[0].item()].unsqueeze(0))

        tri_loss = tri_loss / (logits.shape[0])

        # During incremental learning phases.
        if exemplar is not None:
            target_labels = []

            proto = model.module.fc.weight.detach().cpu()

            for class_index in range(args.base_class):
                feature = torch.reshape(exemplar[class_index, :],
                                        [save_sample, dim, dim])
                temp = proto[class_index]
                skeleton = torch.reshape(temp, [dim, int(model.module.num_features/dim)])

                result = torch.reshape(torch.matmul(feature, skeleton), [save_sample, model.module.num_features])
                if class_index == 0:
                    recovery = result
                else:
                    recovery = torch.cat((recovery, result), 0)
                target_labels += save_sample * [class_index]

            for class_index in range(args.base_class, (args.base_class+args.way*(session - 1))):
                feature = torch.reshape(exemplar[class_index, :],
                                        [save_sample, dim, dim])

                feature = feature[0:args.shot, :]

                temp = proto[class_index]
                skeleton = torch.reshape(temp, [dim, int(model.module.num_features/dim)])

                result = torch.reshape(torch.matmul(feature, skeleton), [args.shot, model.module.num_features])

                recovery = torch.cat((recovery, result), 0)
                target_labels += args.shot * [class_index]

            recovery = recovery.float().cuda()

            # Combine recovered samples with current samples
            logits = torch.cat((recovery, logits), 0)
            logits = F.linear(F.normalize(logits, p=2, dim=-1), F.normalize(model.module.fc.weight, p=2, dim=-1))
            logits = args.temperature * logits

            train_label = target_labels + train_label.tolist()
            train_label = np.array(train_label)
            train_label = torch.from_numpy(train_label).long().cuda()

        else:
            logits = output[:, :(args.base_class + args.way*session)]

        loss = F.cross_entropy(logits, train_label)
        loss = loss + tri_loss

        acc = count_acc(logits, train_label)

        if scheduler is not None:
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, loss.item(), acc))
        else:
            tqdm_gen.set_description(
                'Session {}, epo {},total loss={:.4f} acc={:.4f}'.format(session, epoch, loss.item(), acc))

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args, save_samples, dim=4, exemplar=None):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []

    if exemplar == None:
        exemplar = torch.zeros([args.num_classes, save_samples, dim * dim])
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            _, embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        class_mean = embedding_this.mean(0)
        proto_list.append(class_mean)

        cos_sim = model.module.cos_similarity(class_mean, embedding_this)
        save_far_elems = embedding_this[
            torch.topk(cos_sim, int(save_samples / 2), largest=False, sorted=False).indices]
        save_close_elems = embedding_this[
            torch.topk(cos_sim, int(save_samples / 2), largest=True, sorted=False).indices]
        save_far_elems = save_far_elems.view(int(save_samples / 2), dim,
                                             int(model.module.num_features / dim))
        save_close_elems = save_close_elems.view(int(save_samples / 2), dim,
                                                 int(model.module.num_features / dim))

        for item in range(int(save_samples / 2)):
            Qf, Rf = torch.linalg.qr(save_far_elems[item, :, :], mode="complete")
            Qc, Rc = torch.linalg.qr(save_close_elems[item, :, :], mode="complete")
            exemplar[class_index, 2 * item, :] = torch.reshape(Qf, [dim * dim])
            exemplar[class_index, 2 * item + 1, :] = torch.reshape(Qc, [dim * dim])

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model, exemplar



def test(model, testloader, epoch,args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits, _ = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

