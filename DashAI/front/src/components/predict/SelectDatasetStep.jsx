/*
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
import { getDatasets as getDatasetsRequest } from "../../api/datasets";
import { formatDate } from "../../utils";

const columns = [
  {
    field: "name",
    headerName: "Name",
    minWidth: 250,
    editable: false,
  },
  {
    field: "created",
    headerName: "Created",
    minWidth: 200,
    type: Date,
    valueFormatter: (params) => formatDate(params.value),
    editable: false,
  },
];

function SelectDatasetStep({ selectedDatasetId, setSelectedDatasetId, setNextEnabled }) {
  const [datasets, setDatasets] = useState([]);
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    async function fetchDatasets() {
      try {
        const response = await getDatasetsRequest();
        setDatasets(response.data);
      } catch (error) {
        enqueueSnackbar("Failed to fetch datasets", { variant: "error" });
      }
    }

    fetchDatasets();
  }, [enqueueSnackbar]);

  const handleRowClick = (params) => {
    setSelectedDatasetId(params.id);
    setNextEnabled(true);
  };

  return (
    <Paper sx={{ height: 400, width: '100%' }}>
      <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
        Select a Dataset
      </Typography>
      <DataGrid
        rows={datasets}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5]}
        onRowClick={handleRowClick}
        getRowId={(row) => row.id}
      />
    </Paper>
  );
}

SelectDatasetStep.propTypes = {
  selectedDatasetId: PropTypes.string,
  setSelectedDatasetId: PropTypes.func.isRequired,
  setNextEnabled: PropTypes.func.isRequired,
};

export default SelectDatasetStep;

*/

import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";

import {
  Alert,
  AlertTitle,
  Grid,
  Link,
  Paper,
  Typography,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import { useSnackbar } from "notistack";
import { Link as RouterLink } from "react-router-dom";

import { getDatasets as getDatasetsRequest } from "../../api/datasets";
import { formatDate } from "../../utils";

const columns = [
  {
    field: "name",
    headerName: "Name",
    minWidth: 250,
    editable: false,
  },
  {
    field: "created",
    headerName: "Created",
    minWidth: 200,
    type: Date,
    valueFormatter: (params) => formatDate(params.value),
    editable: false,
  },
  {
    field: "last_modified",
    headerName: "Last modified",
    minWidth: 200,
    type: Date,
    valueFormatter: (params) => formatDate(params.value),
    editable: false,
  },
];

function SelectDatasetStep({
  selectedDatasetId,
  setSelectedDatasetId,
  setNextEnabled,
}) {
  const { enqueueSnackbar } = useSnackbar();

  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [datasetsSelected, setDatasetsSelected] = useState([]);
  const [requestError, setRequestError] = useState(false);

  const getDatasets = async () => {
    setLoading(true);
    try {
      const rawdata = await getDatasetsRequest();
      const datasets = rawdata.filter(
        (dataset) =>
          dataset.for_prediction == true && dataset.prediction_status == false,
      );
      setDatasets(datasets);
    } catch (error) {
      enqueueSnackbar("Error while trying to obtain the datasets list.");
      setRequestError(true);

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
  // fetch datasets when the component is mounting
  useEffect(() => {
    getDatasets();
  }, []);

  useEffect(() => {
    if (datasetsSelected.length > 0) {
      // the index of the table start with 1!
      // const dataset = datasets[datasetsSelected[0] - 1];
      const selectedDatasetId = datasetsSelected[0];
      setSelectedDatasetId(selectedDatasetId);
      setNextEnabled(true);
    }
  }, [datasetsSelected]);

  return (
    <React.Fragment>
      {/* Title and new datasets button */}
      <Grid
        container
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: 4 }}
      >
        <Typography variant="subtitle1" component="h3">
          Select a dataset for the selected task
        </Typography>
      </Grid>

      {/* Datasets Table */}

      {datasets.length === 0 && !loading && !requestError && (
        <React.Fragment>
          <Alert severity="warning" sx={{ mb: 2 }}>
            <AlertTitle>There is no datasets available.</AlertTitle>
            Go to{" "}
            <Link component={RouterLink} to="/app/data">
              data tab
            </Link>{" "}
            to upload one first.
          </Alert>
          <Typography></Typography>
        </React.Fragment>
      )}
      <Paper>
        <DataGrid
          rows={datasets}
          columns={columns}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 10,
              },
            },
          }}
          onRowSelectionModelChange={(newRowSelectionModel) => {
            setDatasetsSelected(newRowSelectionModel);
          }}
          rowSelectionModel={datasetsSelected}
          density="compact"
          pageSizeOptions={[10]}
          loading={loading}
          autoHeight
          hideFooterSelectedRowCount
        />
      </Paper>
    </React.Fragment>
  );
}

SelectDatasetStep.propTypes = {
  newExp: PropTypes.shape({
    id: PropTypes.string,
    name: PropTypes.string,
    dataset: PropTypes.object,
    task_name: PropTypes.string,
    input_columns: PropTypes.arrayOf(PropTypes.number),
    output_columns: PropTypes.arrayOf(PropTypes.number),
    splits: PropTypes.shape({
      has_changed: PropTypes.bool,
      is_random: PropTypes.bool,
      training: PropTypes.number,
      validation: PropTypes.number,
      testing: PropTypes.number,
    }),
    step: PropTypes.string,
    created: PropTypes.instanceOf(Date),
    last_modified: PropTypes.instanceOf(Date),
    runs: PropTypes.array,
  }),
  setNewExp: PropTypes.func.isRequired,
  setNextEnabled: PropTypes.func.isRequired,
};
export default SelectDatasetStep;
