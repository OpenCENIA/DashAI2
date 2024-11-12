import api from "./api";

import type { IPredict } from "../types/predict";
const predictEndpoint = "/v1/predict";

export const get_prediction_tab = async (table: string): Promise<object> => {
  const response = await api.get(`${predictEndpoint}/?table=${table}/`);
  return response.data;
};
