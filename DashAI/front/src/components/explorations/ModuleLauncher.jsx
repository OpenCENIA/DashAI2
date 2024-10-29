import React, { useCallback, useState } from "react";
import PropTypes from "prop-types";

import { Button } from "@mui/material";

import { ExplorationsProvider } from "./context";
import { ExplorationsModal } from "./";

/**
 * Component that launches the explorations modal. It creates the context and modal on open.
 * @param {object} props
 * @param {function} props.onClose - Function to call when the modal is closed.
 * @param {number} props.datasetId - The id of the dataset to explore.
 */
function ModuleLauncher({ onClose = () => {}, datasetId }) {
  const [open, setOpen] = useState(false);

  const handleOpenContent = useCallback(() => {
    setOpen(true);
  }, []);

  const handleCloseContent = useCallback(() => {
    setOpen(false);
    onClose();
  }, [onClose]);

  return (
    <React.Fragment>
      <Button variant="contained" color="primary" onClick={handleOpenContent}>
        Explorations
      </Button>

      {/* only create context and modal on open */}
      {open && (
        <ExplorationsProvider datasetId={datasetId}>
          <ExplorationsModal open={open} onClose={handleCloseContent} />
        </ExplorationsProvider>
      )}
    </React.Fragment>
  );
}

ModuleLauncher.propTypes = {
  onClose: PropTypes.func,
  datasetId: PropTypes.number.isRequired,
};

export default ModuleLauncher;
