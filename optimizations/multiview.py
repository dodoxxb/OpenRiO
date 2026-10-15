"""Frozen five-view scoring and coordinate selection. No labels are accepted."""
import numpy as np
import torch


def select(boxes, scores, labels, threshold=.5, maximum=300):
    from engine.data.dataset.mmeval import rbox_iou
    keep = []
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        ids = ids[np.argsort(-scores[ids], kind='stable')]
        overlap = rbox_iou(boxes[ids], boxes[ids])
        active = np.ones(len(ids), bool)
        for i in range(len(ids)):
            if active[i]:
                keep.append(int(ids[i]))
                active[i+1:] &= overlap[i, i+1:] <= threshold
    return np.asarray(sorted(keep, key=lambda i: -scores[i])[:maximum], dtype=int)


def undo_rotation(boxes, k, edge=800):
    b = boxes.copy()
    for _ in range(k % 4):
        x, y = b[:, 0].copy(), b[:, 1].copy()
        b[:, 0], b[:, 1] = edge-y, x
        b[:, 4] += np.pi/2
    b[:, 4] %= np.pi
    return b


@torch.no_grad()
def views_for(model, post, image):
    if image.shape != (1, 3, 800, 800):
        raise ValueError('The validated view convention requires one 800x800 image')

    def predict(x):
        result = post(model(x), torch.tensor([[800, 800]], device=x.device))[0]
        return {k: result[k].detach().cpu().numpy() for k in ['boxes', 'scores', 'labels']}

    views = [predict(image)]
    flipped = predict(image.flip(-1))
    flipped['boxes'][:, 0] = 800-flipped['boxes'][:, 0]
    flipped['boxes'][:, 4] = (-flipped['boxes'][:, 4]) % np.pi
    views.append(flipped)
    for k in (1, 2, 3):
        pred = predict(torch.rot90(image, k, (-2, -1)).contiguous())
        pred['boxes'] = undo_rotation(pred['boxes'], k)
        views.append(pred)
    return views


def variants(views):
    """Return useful methods and their required matched controls.

    The uniform and weighted outputs retain agreement-selected anchor scores,
    labels, identities, count and order. No NMS is rerun after replacement.
    """
    from engine.data.dataset.mmeval import rbox_iou
    if len(views) != 5:
        raise ValueError('Expected original, horizontal flip and three quarter-turn views')
    b = np.concatenate([x['boxes'] for x in views])
    s = np.concatenate([x['scores'] for x in views])
    labels = np.concatenate([x['labels'] for x in views])
    v = np.concatenate([np.full(len(x['scores']), i) for i, x in enumerate(views)])
    quality = np.zeros(len(s))
    groups = {}
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        z = rbox_iou(b[ids], b[ids])
        vv = v[ids]
        support = np.zeros((len(ids), 5))
        matches = np.full((len(ids), 5), -1, int)
        for view in range(5):
            src, dst = np.flatnonzero(vv != view), np.flatnonzero(vv == view)
            if not len(src) or not len(dst):
                continue
            block = z[np.ix_(src, dst)]
            best = block.argmax(1)
            support[src, view] = block[np.arange(len(src)), best]
            matches[src, view] = dst[best]
        quality[ids] = support.sum(1)/4
        groups[int(cls)] = (ids, z, matches, support)
    ranked = s*(.75+.25*quality)
    original, selected = select(b, s, labels), select(b, ranked, labels)

    def part(ids, scores=s, boxes=b):
        return dict(boxes=boxes[ids].copy(), scores=scores[ids].astype(s.dtype), labels=labels[ids].copy())

    result = {'single_view': {k: val.copy() for k, val in views[0].items()},
              'five_view_nms': part(original), 'fixed_box_rank': part(original, ranked),
              'agreement': part(selected, ranked)}
    replacements = {'uniform': [], 'weighted': []}
    for anchor in selected:
        ids, z, matches, support = groups[int(labels[anchor])]
        i = int(np.searchsorted(ids, anchor))
        members = np.concatenate(([i], matches[i][(matches[i] >= 0) & (support[i] >= .55)]))
        choices = {k: anchor for k in replacements}
        if len(members) >= 3:
            sub = z[np.ix_(members, members)]
            weights = s[ids[members]].astype(float)
            weights /= weights.sum()
            choices['uniform'] = ids[members[int(sub.mean(1).argmax())]]
            choices['weighted'] = ids[members[int((sub @ weights).argmax())]]
        for k in replacements:
            replacements[k].append(choices[k])
    for name, rep in replacements.items():
        rep = np.asarray(rep, int)
        assert np.array_equal(labels[rep], labels[selected])
        result[name] = dict(boxes=b[rep].copy(), scores=ranked[selected].astype(s.dtype),
                            labels=labels[selected].copy())
    return result
