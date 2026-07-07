import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_RMM_WEIGHTS = (0.4, 0.3, 0.3)
EPS = 1e-9


def l2_normalize_np(x, eps=EPS):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def cosine_np(a, b):
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return np.sum(a * b, axis=1)


def feature_entropy(features):
    features = np.asarray(features, dtype=np.float32)
    shifted = features - np.max(features, axis=1, keepdims=True)
    exp_features = np.exp(shifted)
    probs = exp_features / (np.sum(exp_features, axis=1, keepdims=True) + EPS)
    return float(-np.mean(np.sum(probs * np.log(probs + EPS), axis=1)))


def redundancy_score(sim_matrix):
    return float(np.mean(np.std(sim_matrix, axis=1)))


def multiclass_metrics(labels, preds):
    """
    Recognition metrics from predicted class labels and ground-truth labels.
    These scores are different from candidate-box detection Precision/Recall.
    """
    return {
        "precision_weighted": precision_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "precision_micro": precision_score(
            labels, preds, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            labels, preds, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(
            labels, preds, average="micro", zero_division=0
        ),
        "precision_macro": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, preds, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            labels, preds, average="macro", zero_division=0
        ),
        "precision_per_class": precision_score(
            labels, preds, average=None, zero_division=0
        ),
        "recall_per_class": recall_score(
            labels, preds, average=None, zero_division=0
        ),
        "f1_per_class": f1_score(
            labels, preds, average=None, zero_division=0
        ),
    }


def box_iou(box_a, box_b):
    """
    IoU for boxes in (x1, y1, x2, y2) format.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return float(inter_area / (union + EPS))


def candidate_box_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Candidate-box detection metrics used for Table-style Precision/Recall.

    pred_boxes and gt_boxes are lists of (x1, y1, x2, y2). A predicted box is
    counted as TP when it matches one previously unmatched ground-truth box
    with IoU >= iou_threshold. Unmatched predictions are FP, and unmatched
    ground-truth boxes are FN.
    """
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    matched_gt = set()
    true_positive = 0
    false_positive = 0

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            true_positive += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positive += 1

    false_negative = len(gt_boxes) - len(matched_gt)
    precision = true_positive / (true_positive + false_positive + EPS)
    recall = true_positive / (true_positive + false_negative + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "num_pred_boxes": int(len(pred_boxes)),
        "num_gt_boxes": int(len(gt_boxes)),
        "iou_threshold": float(iou_threshold),
    }


def mcg_components_from_fusion_chain(img_feat, seq_feat, fused_feat):
    """
    Operational MCG chain for released-code evaluation:
    1) normalize modality-aware representations to a shared scale,
    2) estimate the redundant 1D-2D overlap,
    3) compare the fused representation with the modality anchor,
    4) report the residual complementary gain.

    The redundancy-aware representations are produced before this metric by
    cross-attention and MI-guided importance updating in the model.
    """
    img_feat = l2_normalize_np(img_feat)
    seq_feat = l2_normalize_np(seq_feat)
    fused_feat = l2_normalize_np(fused_feat)

    modal_anchor = l2_normalize_np((img_feat + seq_feat) * 0.5)
    fusion_gain = cosine_np(fused_feat, modal_anchor)
    redundancy_baseline = np.maximum(cosine_np(img_feat, seq_feat), 0.0)
    complementary_gain = np.maximum(fusion_gain - redundancy_baseline, 0.0)

    return {
        "fusion_gain": float(np.mean(fusion_gain)),
        "redundancy_baseline": float(np.mean(redundancy_baseline)),
        "complementary_gain": float(np.mean(complementary_gain)),
    }


def multiclass_auc(labels, probs, num_classes):
    try:
        return float(roc_auc_score(
            labels,
            probs,
            labels=np.arange(num_classes),
            multi_class="ovr",
            average="weighted",
        ))
    except ValueError:
        return float("nan")


def recall_at_k(sim_matrix, labels, k):
    """
    Category-level R@K.
    A query is correct if at least one top-K result has the same class label.
    This is different from candidate-box Recall and paired-sample retrieval.
    """
    correct = 0
    labels = np.asarray(labels)
    for i in range(sim_matrix.shape[0]):
        topk = np.argsort(sim_matrix[i])[::-1][:k]
        if labels[i] in labels[topk]:
            correct += 1
    return correct / len(labels)


def modality_complementarity_gain(img_feat, seq_feat, fused_feat):
    """
    MCG is computed after common-scale normalization and redundancy removal.
    It measures the effective complementary gain of fused features over
    single-modality/redundant baselines, not raw 1D/2D feature subtraction.
    Inputs should be modality-aware representations produced by the fusion
    module, not unaligned raw heterogeneous features.
    """
    return mcg_components_from_fusion_chain(
        img_feat, seq_feat, fused_feat
    )["complementary_gain"]


def missing_modality_robustness(clean_probs, missing_probs,
                                clean_feat, missing_feat, labels,
                                weights=DEFAULT_RMM_WEIGHTS):
    """
    R@MM is a normalized robustness ratio under missing modality.
    It jointly retains recognition correctness, confidence, and feature
    response stability, then reports missing/full retention.
    """
    return missing_modality_evaluation(
        clean_probs, missing_probs, clean_feat, missing_feat, labels, weights
    )["rmm"]


def missing_modality_evaluation(clean_probs, missing_probs,
                                clean_feat, missing_feat, labels,
                                weights=DEFAULT_RMM_WEIGHTS):
    """
    Dedicated missing-modality evaluation.
    R@MM is not only Acc_missing / Acc_full; the final score combines
    correctness retention, confidence retention, and feature-response stability.
    """
    labels = np.asarray(labels)
    clean_pred = np.argmax(clean_probs, axis=1)
    missing_pred = np.argmax(missing_probs, axis=1)

    clean_acc = np.mean(clean_pred == labels)
    missing_acc = np.mean(missing_pred == labels)
    acc_retention = missing_acc / (clean_acc + EPS)

    clean_conf = np.mean(clean_probs[np.arange(len(labels)), clean_pred])
    missing_conf = np.mean(missing_probs[np.arange(len(labels)), missing_pred])
    conf_retention = missing_conf / (clean_conf + EPS)

    clean_feat = l2_normalize_np(clean_feat)
    missing_feat = l2_normalize_np(missing_feat)
    feature_retention = np.mean(np.maximum(cosine_np(clean_feat, missing_feat), 0.0))

    score = (
        weights[0] * acc_retention +
        weights[1] * conf_retention +
        weights[2] * feature_retention
    )
    return {
        "rmm": float(np.clip(score, 0.0, 1.0)),
        "accuracy_retention": float(acc_retention),
        "confidence_retention": float(conf_retention),
        "feature_stability": float(feature_retention),
        "clean_accuracy": float(clean_acc),
        "missing_accuracy": float(missing_acc),
    }


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(labels)
    return float(ece)


def summarize_latency_benchmark(batch_latencies, batch_sizes):
    """
    Latency benchmark: single-sample response estimated from timed batches.
    This is reported in milliseconds and is separated from throughput.
    """
    batch_latencies = np.asarray(batch_latencies, dtype=np.float64)
    batch_sizes = np.asarray(batch_sizes, dtype=np.float64)

    if batch_latencies.size == 0 or np.sum(batch_sizes) <= 0:
        return {
            "avg_sample_latency_ms": 0.0,
            "avg_batch_latency_ms": 0.0,
        }

    sample_latencies_ms = batch_latencies * 1000.0 / np.maximum(batch_sizes, 1.0)
    return {
        "avg_sample_latency_ms": float(np.mean(sample_latencies_ms)),
        "avg_batch_latency_ms": float(np.mean(batch_latencies) * 1000.0),
    }


def summarize_throughput_benchmark(batch_latencies, batch_sizes):
    """
    Throughput benchmark: batched processing capacity in samples/s.
    This is not treated as the reciprocal of single-sample latency.
    """
    batch_latencies = np.asarray(batch_latencies, dtype=np.float64)
    batch_sizes = np.asarray(batch_sizes, dtype=np.float64)

    if batch_latencies.size == 0 or np.sum(batch_sizes) <= 0:
        return {
            "total_forward_time": 0.0,
            "throughput": 0.0,
        }

    total_forward_time = float(np.sum(batch_latencies))
    return {
        "total_forward_time": total_forward_time,
        "throughput": float(np.sum(batch_sizes) / (total_forward_time + EPS)),
    }


def summarize_inference_timing(batch_latencies, batch_sizes):
    """
    Summarize latency and throughput separately.
    Single-sample latency and batched throughput are not reciprocal metrics.
    """
    latency = summarize_latency_benchmark(batch_latencies, batch_sizes)
    throughput = summarize_throughput_benchmark(batch_latencies, batch_sizes)
    return {**throughput, **latency}
