import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";

import { Box, CircularProgress, Tooltip, Typography } from "@mui/material";

import { getExplorerResults } from "../../../../api/explorer";
import { TabularVisualizer, PlotlyJsonVisualizer } from "../../Visualizations";

/**
 * NullCell component to render null values in the tabular visualizer
 * @param {Object} props
 */
function NullCell({}) {
  const [hover, setHover] = useState(false);
  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <Typography variant="body2" color="text.disabled">
        {hover ? "None" : "-"}
      </Typography>
    </Box>
  );
}

const visualizers = {
  tabular: TabularVisualizer,
  plotly_json: PlotlyJsonVisualizer,
};
const visualizersKeys = {
  tabular: "tabular",
  plotly_json: "plotly_json",
};

const ORIENTATIONS = {
  dict: "dict",
  records: "records",
};

/**
 * Get the data from the orientation given. This function is used to transform the data
 * from the explorer results to the format required by the tabular visualizer.
 * @param {Object} data The data from the explorer results
 * @param {String} orientation The orientation of the data
 */
const getDataFromOrientation = (data, orientation) => {
  let res = {
    columns: [],
    rows: [],
  };

  if (orientation === ORIENTATIONS.records) {
    throw new Error(`orientation ${orientation} not supported`);
  }

  if (orientation === ORIENTATIONS.dict) {
    // ‘dict’ (default) : dict like {column -> {index -> value}}
    // Get the columns
    const columns = Object.keys(data);
    res.columns = [
      {
        field: "id",
        headerName: "Index",
        width: 150,
      },
      ...columns.map((column) => {
        return {
          field: column,
          headerName: column,
          width: 150,
          renderCell: (params) => {
            if (params.value === null) {
              return <NullCell />;
            } else if (typeof params.value === "object") {
              return (
                <Typography variant="body2" color="text.secondary">
                  {JSON.stringify(params.value)}
                </Typography>
              );
            } else if (
              params.value !== "" &&
              !isNaN(params.value) &&
              !Number.isInteger(params.value)
            ) {
              const tooltip = params.value;
              const display = parseFloat(params.value).toFixed(2);
              return (
                <Tooltip title={tooltip}>
                  <Typography variant="body2">{display}</Typography>
                </Tooltip>
              );
            }
            return <Typography variant="body2">{params.value}</Typography>;
          },
        };
      }),
    ];

    // Get the rows
    const rows = [];
    const indexes = Object.keys(data[columns[0]]);
    indexes.forEach((index) => {
      const row = {
        id: index,
      };
      columns.forEach((column) => {
        row[column] = data[column][index];
      });
      rows.push(row);
    });
    res.rows = rows;
  }

  return res;
};

/**
 * Results component to render the results of the exploration
 * @param {Object} props
 * @param {Number} props.id The id of the exploration
 * @param {Boolean} props.updateFlag Flag to update the results
 * @param {Function} props.setUpdateFlag Function to set the update flag
 */
function Results({ id, updateFlag = false, setUpdateFlag = () => {} }) {
  const [loading, setLoading] = useState(false);
  const [dataType, setDataType] = useState(null);
  const [data, setData] = useState(null);

  const fetchExplorerResults = async () => {
    setLoading(true);
    getExplorerResults(id)
      .then((results) => {
        if (!results?.type) {
          throw new Error("No result type specified in the response");
        }

        // Check if there is an appropriate visualizer
        if (!Object.keys(visualizers).includes(results.type)) {
          throw new Error(`No visualizer found for type: ${results.type}`);
        }
        setDataType(results.type);

        if (results.type === visualizersKeys.tabular) {
          // Get the data from the orientation
          const data = getDataFromOrientation(
            results.data,
            results.config.orient,
          );
          setData({
            columns: data.columns,
            rows: data.rows,
          });
        }

        if (results.type === visualizersKeys.plotly_json) {
          setData(JSON.parse(results.data));
        }
      })
      .catch((error) => {
        console.error(error);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  // Fetch the results data on mount
  useEffect(() => {
    // Fetch the results data
    if (id) {
      fetchExplorerResults();
    }
  }, [id]);

  // Fetch the results data on update flag
  useEffect(() => {
    if (updateFlag && id) {
      fetchExplorerResults();
      setUpdateFlag(false);
    }
  }, [updateFlag]);

  return (
    <Box
      sx={{
        height: "100%",
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {loading && <CircularProgress />}

      {!loading && dataType === visualizersKeys.tabular && (
        <TabularVisualizer
          loading={loading}
          columns={data.columns}
          rows={data.rows}
        />
      )}

      {!loading && dataType === visualizersKeys.plotly_json && (
        <PlotlyJsonVisualizer data={data} />
      )}
    </Box>
  );
}

Results.propTypes = {
  id: PropTypes.number.isRequired,
  updateFlag: PropTypes.bool,
  setUpdateFlag: PropTypes.func,
};

export default Results;
