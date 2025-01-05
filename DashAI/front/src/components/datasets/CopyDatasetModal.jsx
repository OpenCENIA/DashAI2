import React from "react";
import PropTypes from "prop-types";
import { useSnackbar } from "notistack";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";
import { copyDataset } from "../../api/datasets";

function CopyDatasetModal({
  datasetId,
  updateDatasetId,
  open,
  setOpen,
  modifyDataset,
}) {
  const { enqueueSnackbar } = useSnackbar();

  const handleDatasetModification = async (id) => {
    modifyDataset(id);
    setOpen(false);
  };

  const handleDatasetCopyModification = async () => {
    try {
      const datasetCopy = await copyDataset({
        dataset_id: datasetId,
      });
      updateDatasetId(datasetCopy.id);
      enqueueSnackbar("Dataset copied successfully", {
        variant: "success",
      });

      handleDatasetModification(datasetCopy.id);
    } catch (error) {
      enqueueSnackbar("Error while trying to create a copy of the dataset.");
    } finally {
      setOpen(false);
    }
  };
  return (
    <Dialog open={open} onClose={() => setOpen(false)}>
      <DialogTitle>Existing experiments</DialogTitle>
      <DialogContent>
        <DialogContentText>
          This dataset is currently used in existing experiments. Modifying it
          may impact the results of re-running those experiments. Would you like
          to create and modify a copy of this dataset instead?
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setOpen(false)} autoFocus>
          Cancel
        </Button>
        <Button onClick={() => handleDatasetModification(datasetId)}>
          Modify anyway
        </Button>
        <Button
          onClick={handleDatasetCopyModification}
          variant="contained"
          color="primary"
        >
          Make a copy
        </Button>
      </DialogActions>
    </Dialog>
  );
}
CopyDatasetModal.propTypes = {
  datasetId: PropTypes.number.isRequired,
  updateDatasetId: PropTypes.func.isRequired,
  open: PropTypes.bool.isRequired,
  setOpen: PropTypes.func.isRequired,
  modifyDataset: PropTypes.func.isRequired,
};

export default CopyDatasetModal;
