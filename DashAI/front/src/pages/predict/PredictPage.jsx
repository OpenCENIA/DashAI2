import React, { useState } from "react";
import CustomLayout from "../../components/custom/CustomLayout";
import PredictionTable from "../../components/predict/PredictionTable";
import PredictModal from "../../components/predict/PredictModal";
function PredictPage() {
  const [updateTableFlag, setUpdateTableFlag] = useState(false);
  const [handleNewPredict, setHandleNewPredict] = useState(false);

  return (
    <CustomLayout
      title="Prediction Module"
      subtitle="Use a model to make predictions"
    >
      {/* Trained models table */}
      <PredictionTable
        updateTableFlag={updateTableFlag}
        setUpdateTableFlag={setUpdateTableFlag}
        handleNewPredict={() => setHandleNewPredict(true)}
      />

      <PredictModal
        open={handleNewPredict}
        onClose={() => setHandleNewPredict(false)}
        updatePredictions={() => setUpdateTableFlag(true)}
      />
    </CustomLayout>
  );
}

export default PredictPage;
