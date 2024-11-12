import React, { useEffect, useState } from "react";
import { DataGrid, GridActionsCellItem, GridToolbar } from "@mui/x-data-grid";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "notistack";
import { Button, Grid, Paper, Typography } from "@mui/material";
import { get_prediction_tab } from "../../api/predict";
import { formatDate } from "../../utils";
import {
  AddCircleOutline as AddIcon,
  Update as UpdateIcon,
} from "@mui/icons-material";
import DatasetSummaryModal from "../datasets/DatasetSummaryModal";

function PredictionTable({
  handleNewPredict,
  updateTableFlag,
  setUpdateTableFlag,
}) {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState([]);

  const columns = React.useMemo(() => [
    {
      field: "id",
      headerName: "ID",
      minWidth: 30,
      editable: false,
    },
    {
      field: "dataset_name",
      headerName: "Dataset Name",
      minWidth: 200,
      editable: false,
    },
    {
      field: "model_name",
      headerName: "Model Name",
      minWidth: 200,
      editable: false,
    },
    {
      field: "run_name",
      headerName: "Model",
      minWidth: 170,
      editable: false,
    },

    {
      field: "task_name",
      headerName: "Task",
      minWidth: 200,
      editable: false,
    },
    {
      field: "last_modified",
      headerName: "Created",
      minWidth: 170,
      editable: false,
      type: Date,
      valueFormatter: (params) => formatDate(params.value),
    },

    {
      field: "actions",
      type: "actions",
      minWidth: 30,
      getActions: (params) => [
        <DatasetSummaryModal
          key="dataset-summary-component"
          datasetId={params.id}
        />,
      ],
    },
  ]);
  const tabla = "PredictionTable";
  const getModels = async (table) => {
    setLoading(true);
    try {
      const models = await get_prediction_tab(table);
      const uniqueModels = Array.from(
        new Set(models.map((model) => model.id)),
      ).map((id) => models.find((model) => model.id === id));
      setModels(uniqueModels);
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
    getModels(tabla);
  }, []);

  // triggers an update of the table when updateFlag is set to true
  useEffect(() => {
    if (updateTableFlag) {
      setUpdateTableFlag(false);
      getModels();
    }
  }, [updateTableFlag]);

  return (
    <Paper sx={{ py: 4, px: 6 }}>
      {/* Title and new datasets button */}
      <Grid
        container
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: 4 }}
      >
        <Typography variant="h5" component="h2">
          Current predicted datasets
        </Typography>
        <Grid item>
          <Grid container spacing={2}>
            <Grid item>
              <Button
                variant="contained"
                onClick={handleNewPredict}
                endIcon={<AddIcon />}
              >
                New Prediction
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                onClick={() => setUpdateTableFlag(true)}
                endIcon={<UpdateIcon />}
              >
                Update
              </Button>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Datasets Table */}
      <DataGrid
        rows={models}
        columns={columns}
        initialState={{
          pagination: {
            paginationModel: {
              pageSize: 5,
            },
          },
        }}
        pageSize={5}
        sortModel={[{ field: "id", sort: "asc" }]}
        pageSizeOptions={[5, 10]}
        disableRowSelectionOnClick
        autoHeight
        loading={loading}
        slots={{
          toolbar: GridToolbar,
        }}
      />
    </Paper>
  );
}

export default PredictionTable;
