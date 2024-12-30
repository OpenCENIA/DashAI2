import React, { useState } from "react";

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

import PropTypes from "prop-types";
import CloseIcon from "@mui/icons-material/Close";
import { useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";
import SelectModelStep from "./SelectModelStep";
import SelectDatasetStep from "./SelectDatasetStep";
import PredictForm from "./EnqueuePrediction";

function PredictionModal({ open, onClose, updatePredictions }) {
  const theme = useTheme();
  const matches = useMediaQuery(theme.breakpoints.down("md"));
  const screenSm = useMediaQuery(theme.breakpoints.down("sm"));
  const [activeStep, setActiveStep] = useState(0);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [nextEnabled, setNextEnabled] = useState(false);
  const [predictName, setPredictName] = useState("");
  const [trainDataset, setTrainDataset] = useState(null);

  const steps = ["Select Model", "Select Dataset"];

  const handleCloseDialog = () => {
    setActiveStep(0);
    onClose(false);
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

  const handleNextButton = () => {
    if (activeStep === steps.length) {
      handleCloseDialog();
      return;
    }

    setActiveStep((prevStep) => prevStep + 1);
    setNextEnabled(false);
  };

  const handlePredictNameInput = (name) => {
    setPredictName(name);
  };

  return (
    <Dialog
      open={open}
      fullScreen={screenSm}
      fullWidth
      maxWidth={"lg"}
      onClose={handleCloseDialog}
      aria-labelledby="new-predict-dialog-title"
      aria-describedby="new-predict-dialog-description"
      scroll="paper"
      PaperProps={{
        sx: { minHeight: "80vh" },
      }}
    >
      <DialogTitle>
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
                  Create a New Prediction
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
                  key={`${step}`}
                  completed={activeStep > index}
                  disabled={activeStep < index}
                >
                  <StepButton color="inherit" onClick={handleStepButton(index)}>
                    {step}
                  </StepButton>
                </Step>
              ))}
            </Stepper>
          </Grid>
        </Grid>
      </DialogTitle>

      <DialogContent dividers>
        {activeStep === 0 && (
          <SelectModelStep
            setSelectedModelId={setSelectedModelId}
            setNextEnabled={setNextEnabled}
            onPredictNameInput={handlePredictNameInput}
            setTrainDataset={setTrainDataset}
          />
        )}
        {activeStep === 1 && (
          <SelectDatasetStep
            setSelectedDatasetId={setSelectedDatasetId}
            setNextEnabled={setNextEnabled}
            trainDataset={trainDataset}
          />
        )}
        {activeStep === 2 && (
          <PredictForm
            run_id={selectedModelId}
            id={selectedDatasetId}
            json_filename={predictName}
            onClose={handleCloseDialog}
            updatePredictions={updatePredictions}
          />
        )}
      </DialogContent>

      <DialogActions>
        <ButtonGroup size="large">
          <Button onClick={handleBackButton}>
            {activeStep === 0 ? "Close" : "Back"}
          </Button>
          <Button
            onClick={handleNextButton}
            autoFocus
            variant="contained"
            color="primary"
            disabled={!nextEnabled}
          >
            {activeStep === 1 ? "Save" : "Next"}
          </Button>
        </ButtonGroup>
      </DialogActions>
    </Dialog>
  );
}

PredictionModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  updatePredictions: PropTypes.func.isRequired,
};

export default PredictionModal;
