import axios from "axios";

const API_BASE = "http://localhost:8000";

export const extractText = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await axios.post(`${API_BASE}/extract`, formData);
    return res.data.text;
};

export const summarizeBert = async (text) => {
    const res = await axios.post(`${API_BASE}/bert`, { text });
    return res.data;
};

export const summarizeKrylov = async (text) => {
    const res = await axios.post(`${API_BASE}/krylov`, { text });
    return res.data;
};

export const compareSummaries = async (cand, ref) => {
    const res = await axios.post(`${API_BASE}/compare`, { cand, ref });
    return res.data;
};
