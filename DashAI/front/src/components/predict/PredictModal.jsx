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

import CloseIcon from "@mui/icons-material/Close";
import { useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useSnackbar } from "notistack";
import SelectModelStep from "./SelectModelStep";
import SelectDatasetStep from "./SelectDatasetStep";
import PredictForm from "./EnqueuePrediction";

function PredictModal({
  open,
  onClose,
  modelId,
  datasetId,
  updatePredictions,
}) {
  const theme = useTheme();
  const matches = useMediaQuery(theme.breakpoints.down("md"));
  const screenSm = useMediaQuery(theme.breakpoints.down("sm"));
  const [activeStep, setActiveStep] = useState(0);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [nextEnabled, setNextEnabled] = useState(false);

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
    /*
  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setNextEnabled(false);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
    setNextEnabled(true);
  };

  const handleFinish = () => {
    onClose();
  };
*/
    setActiveStep((prevStep) => prevStep + 1);
    setNextEnabled(false);
  };
  return (
    <Dialog
      open={open}
      fullScreen={screenSm}
      fullWidth
      maxWidth={"lg"}
      onClose={handleCloseDialog}
      aria-labelledby="new-experiment-dialog-title"
      aria-describedby="new-experiment-dialog-description"
      scroll="paper"
      PaperProps={{
        sx: { minHeight: "80vh" },
      }}
    >
      {/*
      <DialogTitle>
        <Grid container justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Predict Page</Typography>
          <Stepper activeStep={activeStep}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </Grid>
      </DialogTitle>
*/}

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
                  New predict
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

      <DialogContent dividers>
        {activeStep === 0 && (
          <SelectModelStep
            selectedModelId={selectedModelId}
            setSelectedModelId={setSelectedModelId}
            setNextEnabled={setNextEnabled}
          />
        )}
        {activeStep === 1 && (
          <SelectDatasetStep
            selectedDatasetId={selectedDatasetId}
            setSelectedDatasetId={setSelectedDatasetId}
            setNextEnabled={setNextEnabled}
          />
        )}
        {activeStep === 2 && (
          <PredictForm run_id={selectedModelId} id={selectedDatasetId} />
        )}
      </DialogContent>
      {/*
      <Grid container justifyContent="flex-end" spacing={2} sx={{ p: 2 }}>
        {activeStep > 0 && (
          <Grid item>
            <Button onClick={handleBack}>Back</Button>
          </Grid>
        )}
        {activeStep < steps.length - 1 && (
          <Grid item>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={!nextEnabled}
            >
              Next
            </Button>
          </Grid>
        )}
        {activeStep === steps.length - 1 && (
          <Grid item>
            <Button variant="contained" onClick={handleFinish}>
              Finish
            </Button>
          </Grid>
        )}
      </Grid>
    </Dialog>
  );
}
*/}

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

export default PredictModal;
