import React, { useState } from "react";
import PredictForm from "./EnqueuePrediction";
import PredictionModal from "./PredictionModal";

function PredictionHandler({ setUpdateTableFlag }) {
  const [handleNewPredict, setHandleNewPredict] = useState(false);

  return (
    <PredictionModal
      open={handleNewPredict}
      onClose={() => setHandleNewPredict(false)}
      updatePredictions={() => setUpdateTableFlag(true)}
    />
  );
}

export default PredictionHandler;
