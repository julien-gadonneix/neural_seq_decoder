import re
import pickle
import numpy as np

import torch
from neural_decoder.dataset import SpeechDataset


from neural_decoder.neural_decoder_trainer import getDatasetLoadersSentences
from neural_decoder.neural_decoder_trainer import loadModel
import pickle

modelPath = "/home/jgadonneix/Bureau/PhD/data/willettcontinuous2023/model_logs/speech_logs/speechBaseline4"


with open(modelPath + "/args", "rb") as handle:
    args = pickle.load(handle)
idx = 1
args["datasetPath"] = (
    f"/home/jgadonneix/Bureau/PhD/data/willettcontinuous2023/model_logs/ptDecoder_ctc_sentences_{idx}"
)
loadedData = getDatasetLoadersSentences(args["datasetPath"])

device = "cuda"
model = loadModel(modelPath, device=device)


model.eval()

rnn_outputs = {
    "logits": [],
    "logitLengths": [],
    "trueSeqs": [],
    "transcriptions": [],
}

partition = "sentences"  # "test"
testDayIdxs = range(len(loadedData[partition]))

for i, testDayIdx in enumerate(testDayIdxs):
    print(f"Processing day {i+1} / {len(testDayIdxs)}")
    test_ds = SpeechDataset([loadedData[partition][i]])
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )
    decoderOutput = []
    for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            torch.tensor([testDayIdx + idx * 12], dtype=torch.int64).to(device),
        )
        pred = model.forward(X, dayIdx)
        # print(pred.shape, X.shape)
        # Place predictions at strided positions and fill rest with NaNs
        # pred shape: (batch, O, n_classes+1), X shape: (batch, L, neural_dim)
        # Account for kernel length offset and place at correct strided positions
        pred_expanded = torch.full(
            (pred.shape[0], X.shape[1], pred.shape[2]), float("nan"), device=pred.device
        )
        # Place predictions at strided positions starting after kernel length
        # Formula: output position i corresponds to input position (i * strideLen + kernelLen - 1)
        for k in range(pred.shape[1]):
            input_pos = k * model.strideLen + model.kernelLen - 1
            if input_pos < X.shape[1]:
                pred_expanded[:, input_pos, :] = pred[:, k, :]

        decoderOutput.append(pred_expanded.cpu().detach().numpy().squeeze())
        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        for iterIdx in range(pred.shape[0]):
            trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

            rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
            rnn_outputs["logitLengths"].append(
                adjustedLens[iterIdx].cpu().detach().item()
            )
            rnn_outputs["trueSeqs"].append(trueSeq)

        transcript = loadedData[partition][i]["transcriptions"][j].strip()
        transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
        transcript = transcript.replace("--", "").lower()
        rnn_outputs["transcriptions"].append(transcript)
    decoderOutput = np.concatenate(decoderOutput, axis=0)
    print(decoderOutput.shape)
    with open(
        f"/home/jgadonneix/Bureau/PhD/data/willettcontinuous2023/model_logs/rnn_outputs_day{testDayIdx + idx*12}_sentences",
        "wb",
    ) as handle:
        pickle.dump(decoderOutput, handle)


for i in range(len(rnn_outputs["transcriptions"])):
    new_trans = [ord(c) for c in rnn_outputs["transcriptions"][i]] + [0]
    rnn_outputs["transcriptions"][i] = np.array(new_trans)
