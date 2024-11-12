import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import {
  Alert,
  AlertTitle,
  Grid,
  Paper,
  Typography,
  Button,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import { useSnackbar } from "notistack";
import { get_prediction_tab } from "../../api/predict";
import { formatDate } from "../../utils";

function SelectModelStep({
  selectedModelId,
  setSelectedModelId,
  setNextEnabled,
}) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const { enqueueSnackbar } = useSnackbar();

  const columns = React.useMemo(() => [
    {
      field: "id",
      headerName: "ID",
      minWidth: 30,
      editable: false,
    },
    {
      field: "experiment_name",
      headerName: "Experiment Name",
      minWidth: 170,
      editable: false,
    },

    {
      field: "run_name",
      headerName: "Model Name",
      minWidth: 170,
      editable: false,
    },
    {
      field: "model_name",
      headerName: "Model",
      minWidth: 200,
      editable: false,
    },
    {
      field: "task_name",
      headerName: "Task",
      minWidth: 200,
      editable: false,
    },
    {
      field: "dataset_name",
      headerName: "Dataset Name",
      minWidth: 200,
      editable: false,
    },
    {
      field: "created",
      headerName: "Created",
      minWidth: 170,
      editable: false,
      type: Date,
      valueFormatter: (params) => formatDate(params.value),
    },
  ]);

  const get_Models = async (table) => {
    setLoading(true);
    try {
      const models = await get_prediction_tab(table);
      setModels(models);
    } catch (error) {
      enqueueSnackbar("Error while trying to obtain the models table.");
      if (error.response) {
        console.error("Response error:", error.message);
      } else if (error.request) {
        console.error("Request error", error.request);
      } else {
        console.error("Unknown Error", error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    get_Models("SelectModelStep");
  }, []);

  const handleRowClick = (params) => {
    setSelectedModelId(params.id);
    setNextEnabled(true);
  };

  return (
    <Paper sx={{ height: 400, width: "100%" }}>
      <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
        Select a Model
      </Typography>
      <DataGrid
        rows={models}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5]}
        onRowClick={handleRowClick}
        getRowId={(row) => row.id}
      />
    </Paper>
  );
}

SelectModelStep.propTypes = {
  selectedModelId: PropTypes.string,
  setSelectedModelId: PropTypes.func.isRequired,
  setNextEnabled: PropTypes.func.isRequired,
};

export default SelectModelStep;
