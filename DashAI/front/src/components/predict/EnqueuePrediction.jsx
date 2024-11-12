import React, { useEffect, useRef, useState } from "react";
import api from "../../api/api";
import {
  TextField,
  Button,
  Typography,
  Box,
  CircularProgress,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "notistack";
import {
  enqueuePredictionJob as enqueuePredictionRequest,
  startJobQueue as startJobQueueRequest,
} from "../../api/job";
import { getRunStatus } from "../../utils/runStatus";

const PredictForm = ({ run_id, id }) => {
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const intervalRef = useRef(null);

  const enqueuePredictionJob = async (run_id, id) => {
    try {
      await enqueuePredictionRequest(run_id, id);
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
    } catch (error) {
      console.error("Error starting job queue:", error);
      enqueueSnackbar("Error starting job queue", { variant: "error" });
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const error = await enqueuePredictionJob(run_id, id);
      if (!error) {
        enqueueSnackbar("Prediction job enqueued successfully", {
          variant: "success",
        });
        await startJobQueue();
      }
    } catch (err) {
      console.error("Error:", err);
      enqueueSnackbar("Error submitting prediction job", { variant: "error" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (run_id && id) {
      handleSubmit();
    }
  }, [run_id, id]);
  return (
    <div>
      {loading && <p>Loading...</p>}
      {/* Add any other UI elements you need */}
    </div>
  );
};

export default PredictForm;
