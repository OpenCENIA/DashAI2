import React, { useState } from "react";
import PredictForm from "./EnqueuePrediction";
import PredictModal from "../../components/predict/PredictModal";

function PredictionHandler({ updateTableFlag, setUpdateTableFlag }) {
  const [handleNewPredict, setHandleNewPredict] = useState(false);
  const [modelId, setModelId] = useState(null);
  const [datasetId, setDatasetId] = useState(null);

  const openPredictModal = (modelId, datasetId) => {
    setModelId(modelId);
    setDatasetId(datasetId);
    setHandleNewPredict(true);
  };

  return (
    <>
      {/* PredictModal */}
      <PredictModal
        open={handleNewPredict}
        onClose={() => setHandleNewPredict(false)}
        modelId={modelId}
        datasetId={datasetId}
        updatePredictions={() => setUpdateTableFlag(true)}
      />

      {/* Prediction form */}
      <PredictForm run_id={modelId} id={datasetId} />
    </>
  );
}

export default PredictionHandler;
