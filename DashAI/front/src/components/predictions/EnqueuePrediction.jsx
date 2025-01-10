import React, { useEffect, useRef, useState } from "react";
import { useSnackbar } from "notistack";
import {
  enqueuePredictionJob as enqueuePredictionRequest,
  startJobQueue as startJobQueueRequest,
} from "../../api/job";

const EnqueuePrediction = ({
  run_id,
  id,
  json_filename,
  onClose,
  updatePredictions,
}) => {
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);

  const enqueuePredictionJob = async (run_id, id, json_filename) => {
    try {
      await enqueuePredictionRequest(run_id, id, json_filename);
      onClose();
      return false;
    } catch (error) {
      enqueueSnackbar(
        `Error enqueuing prediction with runId, id ${run_id}, ${id}`,
        { variant: "error" },
      );
      if (error.response) {
        console.error("Response error:", error.message);
      } else if (error.request) {
        console.error("Request error", error.request);
      } else {
        console.error("Unknown Error", error.message);
      }
      return true; // return true for error
    }
  };

  const startJobQueue = async () => {
    try {
      await startJobQueueRequest();
      updatePredictions();
    } catch (error) {
      console.error("Error starting job queue:", error);
      enqueueSnackbar("Error starting job queue", { variant: "error" });
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const error = await enqueuePredictionJob(run_id, id, json_filename);
      if (!error) {
        enqueueSnackbar("Prediction job enqueued successfully", {
          variant: "success",
        });
        await startJobQueue();
      }
    } catch (err) {
      if (statusCode == 400) {
        enqueueSnackbar("Invalid dataest ", { variant: "error" });
      }
      const statusCode = err.response?.status;
      const errorMessage = err.response?.data?.detail;

      console.error("Error:", err);
      enqueueSnackbar("Error submitting prediction job", { variant: "error" });
    } finally {
      setLoading(false);
    }
  };

  const didRun = useRef(false); // Ref to track if the effect has already been run

  useEffect(() => {
    if (run_id && id && !didRun.current) {
      didRun.current = true; // Set to true to prevent re-running
      handleSubmit();
      updatePredictions();
    }
  });

  return <div>{loading && <p>Loading...</p>}</div>;
};

export default EnqueuePrediction;
