import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";

import { Box, ButtonGroup, Paper, Typography } from "@mui/material";
import { DataGrid, GridToolbar } from "@mui/x-data-grid";

import {
  PlayArrow as PlayArrowIcon,
  Check as CheckIcon,
} from "@mui/icons-material";

import { LoadingButton } from "@mui/lab";
import { useSnackbar } from "notistack";

import { ExplorerStatus } from "../../types/explorer";
import { getExplorersByExplorationId as getExplorersRequest } from "../../api/explorer";
import { useExplorationsContext } from "./context";

import {
  enqueueExplorerJob as enqueueJobRequest,
  startJobQueue as startJobQueueRequest,
} from "../../api/job";
import { formatDate } from "../../utils";

const columns = [
  {
    field: "name",
    headerName: "Name",
    flex: 1,
  },
  {
    field: "exploration_type",
    headerName: "Exploration Type",
    flex: 1,
  },
  {
    field: "status",
    headerName: "Status Value",
    flex: 1,
  },
  {
    field: "status_display",
    headerName: "Status",
    flex: 1,
    valueGetter: (params) => ExplorerStatus[params.row.status],
  },
  {
    field: "last_modified",
    headerName: "Last Modified",
    flex: 1,
    valueFormatter: (params) => formatDate(params.value),
  },
];

/**
 * Component to run explorers from an exploration. It uses context to get the exploration data.
 * @param {Object} props
 * @param {Function} props.handleCloseDialog - Function to close the dialog
 * @param {Boolean} props.updateFlag - Flag to update the explorers
 */
function ExplorationRunner({
  handleCloseDialog = () => {},
  updateFlag = false,
}) {
  const { enqueueSnackbar } = useSnackbar();
  const { explorationData } = useExplorationsContext();
  const { id: explorationId, explorers } = explorationData;

  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState(explorers);
  const [rowSelectionModel, setRowSelectionModel] = useState(
    explorers.map((explorer) => explorer.id),
  ); // Select all explorers by default

  const [running, setRunning] = useState(false);
  const [finishedRunning, setFinishedRunning] = useState(false);

  const launchJob = async (explorerId) => {
    return enqueueJobRequest(explorerId);
  };

  const submitExecutions = async () => {
    return Promise.all(
      rowSelectionModel.map((explorerId) => launchJob(explorerId)),
    );
  };

  const handleExecuteExplorers = async () => {
    setRunning(true);
    // send runs to the job queue
    submitExecutions()
      .then(() => {
        getExplorers();
        startJobQueueRequest(true); // true to stop when queue empties
      })
      .catch((error) => {
        console.log(error);
        enqueueSnackbar("Error while trying to start explorers", {
          variant: "error",
        });
      });
  };

  const getExplorers = async () => {
    setLoading(true);
    getExplorersRequest(explorationId)
      .then((explorers) => {
        setRows(explorers);
      })
      .catch((error) => {
        console.log(error);
        enqueueSnackbar("Error while trying to fetch explorers", {
          variant: "error",
        });
      })
      .finally(() => {
        setLoading(false);
      });
  };

  // update state of explorer jobs
  useEffect(() => {
    if (rows.length > 0) {
      let running = rows.some(
        (explorer) =>
          explorer.status === ExplorerStatus.DELIVERED ||
          explorer.status === ExplorerStatus.STARTED,
      );
      let finished = rows.every(
        (explorer) => explorer.status === ExplorerStatus.FINISHED,
      );
      setRunning(running);
      setFinishedRunning(finished);
    }
  }, [rows]);

  // polling to update the state of the runs
  useEffect(() => {
    if (updateFlag) {
      getExplorers();
    }

    if (running) {
      const interval = setInterval(() => {
        getExplorers();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [running, updateFlag]);

  return (
    <Box
      sx={{
        height: "100%",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        gap: 2,
      }}
    >
      <Paper
        sx={{ display: "flex", flexDirection: "column", px: 3, py: 2 }}
        // solves a mui problem related to putting datagrid inside another datagrid
        onClick={(event) => {
          event.target = document.body;
        }}
      >
        <Typography variant="subtitle1" component="h3" sx={{ pb: 1 }}>
          Select explorers to run
        </Typography>
        <DataGrid
          autoHeight
          loading={loading}
          rows={rows}
          columns={columns}
          checkboxSelection
          onRowSelectionModelChange={(newRowSelectionModel) => {
            setRowSelectionModel(newRowSelectionModel);
          }}
          rowSelectionModel={rowSelectionModel}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 5,
              },
            },
            columns: {
              columnVisibilityModel: {
                status: false,
              },
            },
            sorting: {
              sortModel: [
                {
                  field: "last_modified",
                  sort: "desc",
                },
              ],
            },
          }}
          pageSizeOptions={[5, 10]}
          disableRowSelectionOnClick
          slots={{
            toolbar: GridToolbar,
          }}
          slotProps={{
            toolbar: {
              showQuickFilter: true,
              csvOptions: {
                disableToolbarButton: true,
              },
              printOptions: {
                disableToolbarButton: true,
              },
            },
          }}
          density="compact"
        />
      </Paper>

      <ButtonGroup size="large" sx={{ justifyContent: "flex-end" }}>
        {!running && finishedRunning && (
          <LoadingButton
            variant="outlined"
            loading={running}
            endIcon={<PlayArrowIcon />}
            onClick={handleExecuteExplorers}
            disabled={rowSelectionModel.length === 0}
          >
            Re Run
          </LoadingButton>
        )}

        <LoadingButton
          variant="contained"
          loading={running}
          endIcon={finishedRunning ? <CheckIcon /> : <PlayArrowIcon />}
          onClick={
            finishedRunning ? () => handleCloseDialog() : handleExecuteExplorers
          }
          disabled={!finishedRunning && rowSelectionModel.length === 0}
        >
          {finishedRunning ? "Finished" : "Start"}
        </LoadingButton>
      </ButtonGroup>
    </Box>
  );
}

ExplorationRunner.propTypes = {
  handleCloseDialog: PropTypes.func,
  updateFlag: PropTypes.bool,
};

export default ExplorationRunner;
