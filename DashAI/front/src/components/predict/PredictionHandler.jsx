import React, { useState } from "react";
import PredictForm from "./EnqueuePrediction";
import PredictModal from "../../components/predict/PredictModal";

function PredictionHandler({ setUpdateTableFlag }) {
  const [handleNewPredict, setHandleNewPredict] = useState(false);

  return (
    <PredictModal
      open={handleNewPredict}
      onClose={() => setHandleNewPredict(false)}
      updatePredictions={() => setUpdateTableFlag(true)}
    />
  );
}

export default PredictionHandler;
