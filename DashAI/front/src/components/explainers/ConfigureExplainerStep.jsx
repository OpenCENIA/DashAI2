import React, { useState, useEffect } from "react";
import {
  DialogContentText,
  Grid,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import PropTypes from "prop-types";

import FormSchema from "../shared/FormSchema";
import FormSchemaLayout from "../shared/FormSchemaLayout";
import useSchema from "../../hooks/useSchema";

function ConfigureExplainerStep({
  newExpl,
  setNewExpl,
  setNextEnabled,
  formSubmitRef,
  scope,
}) {
  const { defaultValues } = useSchema({ modelName: newExpl.explainer_name });
  const [error, setError] = useState(false);

  const isParamsEmpty =
    !newExpl.parameters || Object.keys(newExpl.parameters).length === 0;

  function filterFitParameters(explainer) {
    const prefix = "fit_parameter_";
    return Object.keys(explainer).reduce(
      (result, key) => {
        if (key.startsWith(prefix)) {
          result.fitParameters[key.slice(prefix.length)] = explainer[key];
        } else {
          result.parameters[key] = explainer[key];
        }
        return result;
      },
      { parameters: {}, fitParameters: {} },
    );
  }

  const handleUpdateParameters = (values) => {
    if (scope === "Local") {
      const { parameters, fitParameters } = filterFitParameters(values);
      setNewExpl((_) => ({
        ...newExpl,
        parameters: parameters,
        fit_parameters: fitParameters,
      }));
    } else {
      setNewExpl((_) => ({ ...newExpl, parameters: values }));
    }
  };

  useEffect(() => {
    if (isParamsEmpty && Boolean(defaultValues)) {
      if (scope === "Local") {
        const { parameters, fitParameters } =
          filterFitParameters(defaultValues);
        setNewExpl((_) => ({
          ...newExpl,
          parameters: parameters,
          fit_parameters: fitParameters,
        }));
      } else {
        setNewExpl((_) => ({ ...newExpl, parameters: defaultValues }));
      }
    }
  }, [isParamsEmpty, defaultValues]);

  useEffect(() => {
    setNextEnabled(!error);
  }, [error]);

  return (
    <Grid
      container
      direction="row"
      justifyContent="space-around"
      alignItems="stretch"
      spacing={3}
    >
      <Grid item xs={12}>
        <Typography variant="h5" component="h3">
          Configure your Explainer
        </Typography>
      </Grid>
      {/* Configure dataloader parameters */}
      <Grid item xs={12} md={6}>
        <Paper
          variant="outlined"
          sx={{ p: 4, maxHeight: "55vh", overflow: "auto" }}
        >
          <Stack spacing={3}>
            <DialogContentText>Explainer configuration</DialogContentText>

            <FormSchemaLayout>
              <FormSchema
                autoSave
                model={newExpl.explainer_name}
                onFormSubmit={(values) => {
                  handleUpdateParameters(values);
                }}
                setError={setError}
                formSubmitRef={formSubmitRef}
              />
            </FormSchemaLayout>
          </Stack>
        </Paper>
      </Grid>
    </Grid>
  );
}

ConfigureExplainerStep.propTypes = {
  newExpl: PropTypes.shape({
    name: PropTypes.string,
    explainer_name: PropTypes.string,
    dataset_id: PropTypes.number,
    parameters: PropTypes.object,
    fit_parameters: PropTypes.object,
  }),
  setNewExpl: PropTypes.func.isRequired,
  setNextEnabled: PropTypes.func.isRequired,
  formSubmitRef: PropTypes.shape({ current: PropTypes.any }).isRequired,
};

export default ConfigureExplainerStep;
