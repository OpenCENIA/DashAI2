import React, { useState } from "react";
import CustomLayout from "../../components/custom/CustomLayout";
import PredictionTable from "../../components/predictions/PredictionTable";
import PredictionModal from "../../components/predictions/PredictionModal";
function PredictionPage() {
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

      <PredictionModal
        open={handleNewPredict}
        onClose={() => setHandleNewPredict(false)}
        updatePredictions={() => setUpdateTableFlag(true)}
      />
    </CustomLayout>
  );
}

export default PredictionPage;
