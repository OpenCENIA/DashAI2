import api from "./api";

export const getJobs = async (): Promise<object> => {
  const response = await api.get<object>("/v1/job/");
  return response.data;
};

export const enqueueRunnerJob = async (runId: number): Promise<object> => {
  const data = {
    job_type: "ModelJob",
    kwargs: { run_id: runId },
  };

  const response = await api.post<object>("/v1/job/", data);
  return response.data;
};

export const enqueueExplainerJob = async (
  explainerId: number,
  scope: string,
): Promise<object> => {
  const data = {
    job_type: "ExplainerJob",
    kwargs: { explainer_id: explainerId, explainer_scope: scope },
  };

  const response = await api.post<object>("/v1/job/", data);
  return response.data;
};

export const enqueueExplorerJob = async (
  explorerId: number,
): Promise<object> => {
  const data = {
    job_type: "ExplorerJob",
    kwargs: { explorer_id: explorerId },
  };

  const response = await api.post<object>("/v1/job/", data);
  return response.data;
};

export const enqueuePredictionJob = async (
  run_id: number,
  id: number,
  json_filename: string,
): Promise<object> => {
  const data = {
    job_type: "PredictJob",
    kwargs: { run_id: run_id, id: id, json_filename: json_filename },
  };

  const response = await api.post<object>("/v1/job/", data);
  return response.data;
};

export const startJobQueue = async (
  stopWhenQueueEmpties: boolean | undefined,
): Promise<object> => {
  let params = {};

  if (stopWhenQueueEmpties !== undefined) {
    params = { ...params, stop_when_queue_empties: stopWhenQueueEmpties };
  }

  const response = await api.post<object>("/v1/job/start/", null, { params });
  return response.data;
};
