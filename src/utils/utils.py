import dotenv
import pandas as pd

def load_env_vars() -> tuple[str, str, str]:

    DATA_DIR = dotenv.get_key(dotenv.find_dotenv(), 'DATA_DIR')
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), 'MODELS_DIR')
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), 'HISTORIES_DIR')

    return DATA_DIR, MODELS_DIR, HISTORIES_DIR


def history2df(history: dict) -> pd.DataFrame:

    dfs = []
    for key in history.keys():
        df = pd.DataFrame(history[key])
        df['split'] = [key for _ in range(len(df))]
        dfs.append(df)

    return pd.concat(dfs)
