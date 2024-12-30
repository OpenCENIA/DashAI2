import React, { useState } from "react";
import { Search } from "@mui/icons-material";
import PropTypes from "prop-types";
import { GridActionsCellItem } from "@mui/x-data-grid";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  Grid,
  Button,
  DialogActions,
} from "@mui/material";

function PredictionSummaryModal({ predictionId }) {
  const [open, setOpen] = useState(false);
  const handleCloseContent = () => {
    setOpen(false);
  };

  return (
    <>
      <GridActionsCellItem
        key="prediction-summary-button"
        icon={<Search />}
        label="Prediction Summary"
        onClick={() => setOpen(true)}
        sx={{ color: "warning.main" }}
      />
      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        fullWidth
        maxWidth={"md"}
      >
        <DialogTitle>Prediction Summary</DialogTitle>
        <DialogContent>
          <Grid
            container
            direction="row"
            justifyContent="space-around"
            alignItems="stretch"
            spacing={2}
            onClick={(event) => event.stopPropagation()}
          >
            {/* Prediction Summary Table */}
          </Grid>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default PredictionSummaryModal;
