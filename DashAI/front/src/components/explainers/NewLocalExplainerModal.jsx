import React, { useEffect, useState, useRef } from "react";
import PropTypes from "prop-types";
import { useSnackbar } from "notistack";

import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  ButtonGroup,
  Stepper,
  Step,
  StepButton,
  Grid,
  Typography,
  IconButton,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";

import { createLocalExplainer as createLocalExplainerRequest } from "../../api/explainer";
import { enqueueExplainerJob as enqueueExplainerJobRequest } from "../../api/job";
import { startJobQueue as startJobQueueRequest } from "../../api/job";

import ConfigureExplainerStep from "./ConfigureExplainerStep";
import SelectDatasetStep from "./SelectDatasetStep";
import SetNameAndExplainerStep from "./SetNameAndExplainerStep";
import useUpdateFlag from "../../hooks/useUpdateFlag";
import { flags } from "../../constants/flags";
import TimestampWrapper from "../shared/TimestampWrapper";
import { TIMESTAMP_KEYS } from "../../constants/timestamp";
import { LoadingButton } from "@mui/lab";

const steps = [
  { name: "selectExplainer", label: "Set name and explainer" },
  { name: "SelectDataset", label: "Select dataset" },
  { name: "ConfigureExplainer", label: "Configure explainer parameters" },
];

/**
 * This component renders a modal that takes the user through the process of creating a new experiment.
 * @param {bool} open true to open the modal, false to close it
 * @param {function} setOpen function to modify the value of open
 * @param {function} updateExperiments function to update the experiments table
 */
export default function NewLocalExplainerModal({
  open,
  setOpen,
  explainerConfig,
}) {
  const theme = useTheme();
  const matches = useMediaQuery(theme.breakpoints.down("md"));
  const screenSm = useMediaQuery(theme.breakpoints.down("sm"));
  const formSubmitRef = useRef(null);

  const { enqueueSnackbar } = useSnackbar();

  const { runId, taskName } = explainerConfig;

  const defaultNewLocalExpl = {
    name: "",
    run_id: runId,
    explainer_name: null,
    dataset_id: null,
    parameters: null,
    fit_parameters: null,
  };

  const [activeStep, setActiveStep] = useState(0);
  const [nextEnabled, setNextEnabled] = useState(false);
  const [newLocalExpl, setNewLocalExpl] = useState(defaultNewLocalExpl);
  const [isLoading, setIsLoading] = useState(false);

  const { updateFlag: updateExplainers } = useUpdateFlag({
    flag: flags.EXPLAINERS,
  });

  const enqueueLocalExplainerJob = async (explainerId) => {
    try {
      await enqueueExplainerJobRequest(explainerId, "local");
      enqueueSnackbar("Local explainer job successfully created.", {
        variant: "success",
      });
    } catch (error) {
      enqueueSnackbar("Error while trying to enqueue Local explainer job");
      if (error.response) {
        console.error("Response error:", error.message);
      } else if (error.request) {
        console.error("Request error", error.request);
      } else {
        console.error("Unknown Error", error.message);
      }
    }
  };

  const startJobQueue = async () => {
    try {
      await startJobQueueRequest();
    } catch (error) {
      enqueueSnackbar("Error while trying to start job queue");
      if (error.response) {
        console.error("Response error:", error.message);
      } else if (error.request) {
        console.error("Request error", error.request);
      } else {
        console.error("Unknown Error", error.message);
      }
    }
  };

  const uploadNewLocalExplainer = async () => {
    try {
      setIsLoading(true);
      const response = await createLocalExplainerRequest(
        newLocalExpl.name,
        newLocalExpl.run_id,
        newLocalExpl.explainer_name,
        newLocalExpl.dataset_id,
        newLocalExpl.parameters,
        newLocalExpl.fit_parameters,
      );
      const explainerId = response.id;
      await enqueueLocalExplainerJob(explainerId);
      enqueueSnackbar("Local explainer successfully created.", {
        variant: "success",
      });
      await startJobQueue();
      enqueueSnackbar("Running explainer jobs.", {
        variant: "success",
      });
      updateExplainers();
    } catch (error) {
      enqueueSnackbar("Error while trying to create a new explainer");

      if (error.response) {
        console.error("Response error:", error.message);
      } else if (error.request) {
        console.error("Request error", error.request);
      } else {
        console.error("Unknown Error", error.message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseDialog = () => {
    setActiveStep(0);
    setOpen(false);
    setNewLocalExpl(defaultNewLocalExpl);
    setNextEnabled(false);
  };

  const handleStepButton = (stepIndex) => () => {
    setActiveStep(stepIndex);
  };

  const handleBackButton = () => {
    if (activeStep === 0) {
      handleCloseDialog();
    } else {
      setActiveStep(activeStep - 1);
    }
  };

  const handleNextButton = async () => {
    if (activeStep < steps.length - 1) {
      setActiveStep(activeStep + 1);
      setNextEnabled(false);
    } else {
      await uploadNewLocalExplainer();
      handleCloseDialog();
    }
  };

  return (
    <Dialog
      open={open}
      fullScreen={screenSm}
      fullWidth
      maxWidth={"lg"}
      onClose={handleCloseDialog}
      aria-labelledby="new-local-explainer-dialog-title"
      aria-describedby="new-local-explainer-dialog-description"
      scroll="paper"
      PaperProps={{
        sx: { minHeight: "80vh" },
      }}
    >
      {/* Title */}
      <DialogTitle id="new-local-explainer-dialog-title">
        <Grid container direction={"row"} alignItems={"center"}>
          <Grid item xs={12} md={3}>
            <Grid
              container
              direction="row"
              alignItems="center"
              justifyContent="space-between"
            >
              <Grid item xs={1}>
                <IconButton
                  edge="start"
                  color="inherit"
                  onClick={handleCloseDialog}
                  sx={{ display: { xs: "flex", sm: "none" } }}
                >
                  <CloseIcon />
                </IconButton>
              </Grid>
              <Grid item xs={11}>
                <Typography
                  variant="h6"
                  component="h3"
                  align={matches ? "center" : "left"}
                  sx={{ mb: { sm: 2, md: 0 } }}
                >
                  New local explainer
                </Typography>
              </Grid>
            </Grid>
          </Grid>
          <Grid item xs={12} md={9}>
            <Stepper
              nonLinear
              activeStep={activeStep}
              sx={{ maxWidth: "100%" }}
            >
              {steps.map((step, index) => (
                <Step
                  key={`${step.name}`}
                  completed={activeStep > index}
                  disabled={activeStep < index}
                >
                  <StepButton color="inherit" onClick={handleStepButton(index)}>
                    {step.label}
                  </StepButton>
                </Step>
              ))}
            </Stepper>
          </Grid>
        </Grid>
      </DialogTitle>

      {/* Main content - steps */}
      <DialogContent dividers>
        {activeStep === 0 && (
          <SetNameAndExplainerStep
            newExpl={newLocalExpl}
            setNewExpl={setNewLocalExpl}
            setNextEnabled={setNextEnabled}
            scope={"Local"}
            taskName={taskName}
          />
        )}
        {activeStep === 1 && (
          <SelectDatasetStep
            newExpl={newLocalExpl}
            setNewExpl={setNewLocalExpl}
            setNextEnabled={setNextEnabled}
          />
        )}
        {activeStep === 2 && (
          <ConfigureExplainerStep
            newExpl={newLocalExpl}
            setNewExpl={setNewLocalExpl}
            setNextEnabled={setNextEnabled}
            formSubmitRef={formSubmitRef}
            scope={"Local"}
          />
        )}
      </DialogContent>

      {/* Actions - Back and Next */}
      <DialogActions>
        <ButtonGroup size="large">
          <Button onClick={handleBackButton}>
            {activeStep === 0 ? "Close" : "Back"}
          </Button>
          <TimestampWrapper
            eventName={
              activeStep === 1 ? TIMESTAMP_KEYS.explainer.submitLocal : null
            }
          >
            <LoadingButton
              onClick={handleNextButton}
              autoFocus
              variant="contained"
              color="primary"
              disabled={!nextEnabled}
              loading={isLoading}
            >
              {activeStep === 1 ? "Save" : "Next"}
            </LoadingButton>
          </TimestampWrapper>
        </ButtonGroup>
      </DialogActions>
    </Dialog>
  );
}

NewLocalExplainerModal.propTypes = {
  open: PropTypes.bool.isRequired,
  setOpen: PropTypes.func.isRequired,
  explainerConfig: PropTypes.object,
};
