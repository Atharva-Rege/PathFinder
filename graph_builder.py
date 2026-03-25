import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
# from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "required_data"
DEFAULT_SAVE_DIR = MODULE_DIR / "save"


def turn_job_category_to_list(st):
    if type(st) is not str:
        # To facilitate the filtering later
        return ""
    return [x.strip() for x in st.split(";")]


def clean_salary(salary, is_job=False):
    if not is_job:
        salary = str(salary).replace("k", "000").replace("€", "")
        try:
            salary = int(float(salary))
        except ValueError:
            return None
    if salary < 20000:
        return None
    elif 20000 <= salary < 30000:
        return 'cat 1'
    elif 30000 <= salary < 40000:
        return 'cat 2'
    elif 40000 <= salary < 50000:
        return 'cat 3'
    elif 50000 <= salary < 60000:
        return 'cat 4'
    elif 60000 <= salary < 70000:
        return 'cat 5'
    elif 70000 <= salary < 80000:
        return 'cat 6'
    elif 80000 <= salary < 90000:
        return 'cat 7'
    elif 90000 <= salary < 100000:
        return 'cat 8'
    elif 100000 <= salary < 200000:
        return 'cat 9'
    elif salary >= 200000:
        return 'cat 10'
    return None


# def reduce_zip(zipcode):
#     if type(zipcode) is float:
#         if zipcode < 0:
#             return ""
#         zipcode = int(zipcode)
#     if type(zipcode) is not int and (type(zipcode) is not str or not zipcode.isdigit()):
#         return ""
#     zipcode = int(zipcode)
#     if zipcode >= 10000:
#         zipcode = str(zipcode)[0:2]
#     else:
#         zipcode = str(zipcode)[0:1]
#     return zipcode


def get_edge_index(key1_left,
                   key1_right,
                   key2_left,
                   key2_right,
                   data_src_index,
                   data_key1_id,
                   data_key2_id,
                   from_pre=False,
                   with_timestamp=False,
                   timestamp=''
                   ):
    if from_pre:
        # Ensure we keep key2_left column for the second merge
        cols_to_keep = [key1_left]
        if key2_left not in cols_to_keep and key2_left in data_src_index.columns:
            cols_to_keep.append(key2_left)
        if with_timestamp and timestamp in data_src_index.columns and timestamp not in cols_to_keep:
            cols_to_keep.append(timestamp)
        
        # First merge to get mappedID for key1
        tmp_df1 = pd.merge(data_src_index[cols_to_keep], data_key1_id[[key1_right, 'mappedID']], 
                          left_on=key1_left, right_on=key1_right, how='left')
        # Second merge to get mappedID for key2  
        tmp_df2 = pd.merge(tmp_df1, data_key2_id[[key2_right, 'mappedID']], 
                          left_on=key2_left, right_on=key2_right, how='left', suffixes=('_x', '_y')).dropna()

        from_df = torch.LongTensor(([int(i) for i in tmp_df2['mappedID_x']]))
        to_df = torch.LongTensor(([int(i) for i in tmp_df2['mappedID_y']]))

    else:
        # Perform merge to obtain the edges from table 1 to table 2:
        cols_to_select = [key1_left, key2_left]
        if with_timestamp and timestamp in data_src_index.columns:
            cols_to_select.append(timestamp)
        
        tmp_df1 = pd.merge(data_src_index[cols_to_select], data_key1_id, left_on=key1_left, right_on=key1_right, how='left')
        tmp_df2 = pd.merge(tmp_df1, data_key2_id, left_on=key2_left, right_on=key2_right, how='left').dropna()

        from_df = torch.LongTensor(([int(i) for i in tmp_df2['mappedID_x']]))
        to_df = torch.LongTensor(([int(i) for i in tmp_df2['mappedID_y']]))

    if with_timestamp:
        return torch.stack([from_df, to_df], dim=0), tmp_df2[timestamp]

    return torch.stack([from_df, to_df], dim=0)


def str_to_list(s_input):
    if type(s_input) is str:
        s_input = s_input.replace("'", '"')
        if s_input == "None":
            return []
        try:
            return json.loads(s_input)
        except:
            return []
    return []


def load_data(path=DEFAULT_DATA_DIR):
    data_dir = Path(path)

    candidate_skills = data_dir / 'candidate_skills.csv'
    job_candidates = data_dir / 'job_candidates.csv'
    jobs_skills = data_dir / 'jobs_skills.csv'
    candidate = data_dir / 'candidates.csv'
    job = data_dir / 'jobs.csv'
    hierarchy = data_dir / 'hierarchy.csv'
    skills = data_dir / 'hierarchy.csv'
    candidate_pre = data_dir / 'candidate_preprocessing.csv'
    job_pre = data_dir / 'job_preprocessing.csv'
    cities = data_dir / 'cities.csv'
    video_games = data_dir / 'video_games.csv'
    # Some softwares are video games
    video_games = pd.read_csv(video_games)
    video_games = {"http://perso.com/" + x.replace(" ", "_") for x in video_games["itemLabel"]}

    candidate_skills = pd.read_csv(candidate_skills)
    job_candidates = pd.read_csv(job_candidates)


    # cities = pd.read_csv(cities)

    job_skills = pd.read_csv(jobs_skills)
    candidate = pd.read_csv(candidate)
    candidate_pre = pd.read_csv(candidate_pre)
    job = pd.read_csv(job)
    job_pre = pd.read_csv(job_pre)
    job_pre['dateAdded'] = job_pre['dateAdded'].apply(
        lambda x: time.mktime(datetime.strptime(x, "%d/%m/%Y").timetuple()))
    skill = pd.read_csv(skills)
    hierarchy = pd.read_csv(hierarchy)

    # Only consider common skills
    skills_u = set(candidate_skills["skillUri"])
    skills_j = set(job_skills["skillUri"])
    candidate_skills = candidate_skills[[(x in skills_j) and (x not in video_games) for x in candidate_skills["skillUri"]]].reset_index()
    job_skills = job_skills[[(x in skills_u) and (x not in video_games) for x in job_skills["skillUri"]]].reset_index()

    # convert candidate_pre zip to int if possible else None
    # candidate_pre['zip'] = pd.to_numeric(candidate_pre['zip'], errors='coerce')
    # left join candidate_pre and cities on key zip and zip_code
    # candidate_pre = candidate_pre.merge(cities, left_on='zip', right_on='zip_code', how='left')
    # replace k by 000 in salary and remove €
    candidate_pre['salary_current'] = candidate_pre['salary_current'].apply(
        lambda x: str(x).replace("k", "000").replace("€", ""))
    # convert salary to int if possible else None
    candidate_pre['salary_current'] = pd.to_numeric(candidate_pre['salary_current'], errors='coerce').fillna(0)
    # # convert payrate to int if possible else None
    # candidate_pre['daily_rate'] = pd.to_numeric(
    #     candidate_pre['daily_rate'], errors='coerce').fillna(0)
    candidate_pre = candidate_pre.fillna(-1)

    # convert job_pre zip to int if it's possible else None
    # job_pre['zip'] = pd.to_numeric(job_pre['zip'], errors='coerce')
    # left join job_pre and cities on key zip and zip_code
    # job_pre = job_pre.merge(cities, left_on='zip', right_on='zip_code', how='left')
    # replace k by 000 in salary and remove €
    job_pre['salary'] = job_pre['salary'].apply(lambda x: str(x).replace("k", "000").replace("€", ""))
    # convert salary to int if it's possible else None
    job_pre['salary'] = pd.to_numeric(job_pre['salary'], errors='coerce').fillna(0)
    # # convert payrate to int if it's possible else None
    # job_pre['payRate'] = pd.to_numeric(job_pre['payRate'], errors='coerce').fillna(0)

    job_pre = job_pre.fillna(-1)

    # drop duplicates
    job_candidates = job_candidates.drop_duplicates(
        subset=["Job", "name"],
        keep='last').reset_index(drop=True)
    job_candidates["Date Added_caller"] = job_candidates["Date Added_caller"].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    job_candidates["unique_id"] = job_candidates.apply(
        lambda x: str(x['candidate_ID']) + '_' + str(x['job_ID']) + '_' + str(x['timestamp']), axis=1)
    # sort by col timestamp
    job_candidates = job_candidates.sort_values(by=['timestamp'], ascending=True)

    job_company = job[["Job", "JobOrder.clientCorporation"]].drop_duplicates()
    job_company.columns = ['Job', 'company']

    candidate_exp = candidate_pre[["Name", "experience"]].drop_duplicates()
    candidate_exp["experience"] = candidate_exp["experience"].apply(lambda x: x.replace("1-2 ans", "Niveau junior (0-2 ans)")
                                                                                  .replace('2-5 ans', 'intermédiaire (3-6 ans)')
                                                                                  .replace('5-10 ans', 'senior (+6 ans)')
                                                                                  .replace('> 10 ans', 'senior (+6 ans)')
                                                                                  .replace('>10 ans', 'senior (+6 ans)')
                                                                                  .replace('&gt;10 ans', 'senior (+6 ans)')
                                                                                  if isinstance(x, str) else ''
                                                                                 )
    candidate_exp["experience"] = candidate_exp["experience"].apply(lambda x: str(x))
    candidate_exp.columns = ["name", "experience"]

    job_exp = job_pre[["title", "customText6"]].drop_duplicates()
    job_exp['customText6'] = job_exp['customText6'].apply(lambda x: str(x))
    job_exp['customText6'] = job_exp['customText6'].apply(lambda x: x.replace('nan', ''))
    job_exp.columns = ["Job", "experience"]

    candidate_origin = candidate_pre[['Name', "origin"]][
        candidate_pre["origin"] != "None"].drop_duplicates()
    candidate_origin["origin"] = candidate_origin["origin"].apply(
        lambda x: str(x).replace("['", "").replace("']", ""))
    candidate_origin.columns = ["name", "origin"]

    candidate_contract = candidate_pre[["Name", "contract"]].copy()
    candidate_contract.columns = ["name", "contract"]
    candidate_contract["contract"] = candidate_contract["contract"].apply(
        lambda x: x.replace('Permanent', 'CDI').replace('cdi', 'CDI') if isinstance(x, str) else str(x)
    )
    candidate_contract = candidate_contract.drop_duplicates().dropna()

    job_contract = job_pre[["title", "employmentType"]].copy()
    job_contract.columns = ["Job", "contract"]
    job_contract["contract"] = job_contract["contract"].apply(
        lambda x: x.replace('Permanent', 'CDI') if isinstance(x, str) else str(x)
    )


    # candidate_category = candidate_pre[["Name", "category"]].copy()
    # candidate_category.columns = ["name", "category"]
    # candidate_category["category"] = candidate_category["category"].apply(lambda x: str_to_list(x))
    # candidate_category = candidate_category.explode("category").drop_duplicates().dropna()

    job_category = job_pre[["title", "categories"]].copy()
    job_category.columns = ["Job", "category"]
    job_category["category"] = job_category["category"].apply(lambda x: turn_job_category_to_list(x))
    job_category = job_category.explode("category").drop_duplicates().dropna(subset=['category'])
    job_category = job_category[job_category["category"] != ""]

    candidate_pre['salaire_cat'] = candidate_pre['salary_current'].apply(lambda x: clean_salary(x))
    job_pre['salaire_cat'] = job_pre['salary'].apply(lambda x: clean_salary(x, True))

    candidate_salary = candidate_pre[["Name", "salaire_cat", "salary_current"]].drop_duplicates().dropna()
    candidate_salary.columns = ["name", "salaire_cat", 'salaire']

    job_salary = job_pre[["title", "salaire_cat", "salary"]].drop_duplicates().dropna()
    job_salary.columns = ["Job", "salaire_cat", 'salary']

    # candidate_pre["zip_clean"] = candidate_pre["zip"].apply(lambda x: reduce_zip(x))
    # candidate_zip = candidate_pre[["Name", "zip_clean"]].drop_duplicates().dropna()
    # candidate_zip.columns = ["name", "zip"]

    # job_pre["zip_clean"] = job_pre["zip"].apply(lambda x: reduce_zip(x))
    # job_zip = job_pre[["title", "zip_clean"]].drop_duplicates().dropna()
    # job_zip.columns = ["Job", "zip"]

    # Job company (owner field is the company)
    job_company_full = job_pre[["title", "owner"]].drop_duplicates().dropna()
    job_company_full.columns = ["Job", "company"]

    skill_concept = hierarchy[hierarchy['conceptUri'].isin(job_skills['skillUri'])][
        ['conceptUri', 'preferredLabel_parent']]
    skill_concept.columns = ['skillUri', 'concept']

    return candidate, candidate_skills, job_candidates, job_skills, candidate_pre, job, job_pre, skill, hierarchy, candidate_exp, job_exp, candidate_origin, candidate_contract, job_contract, job_category, candidate_salary, job_salary, job_company, job_company_full, skill_concept


def add_nodes_features(unique_nodes_id, source_df, key_match, key_add_feature, key_unique_id):
    convert_dict = source_df[[key_match, key_add_feature]].set_index(key_match).to_dict()[key_add_feature]
    return unique_nodes_id[key_unique_id].apply(lambda x: float(convert_dict[x]) if x in convert_dict.keys() else None)


def load_nodes(
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_pre,
        candidate_exp,
        job_exp,
        candidate_origin,
        candidate_contract,
        job_contract,
        job_category,
        candidate_salary,
        job_salary,
        job_company,
        job_company_full,
        skill_concept,
        job_pre
):
    # Create a mapping from unique user indices to range [0, num_user_nodes):
    # unique_user_id = np.unique(np.append(job_candidates['name'],candidate_skills['name']))
    # unique_user_id = candidate_pre['Name'].unique()

    unique_user_id = candidate_pre['Name']

    unique_user_id = pd.DataFrame.from_dict(data={
        'nameID': unique_user_id,
        'mappedID': list(pd.RangeIndex(len(unique_user_id)))
    })

    # add node data for : timestamp, salary, payrate, latitude, longitude
    unique_user_id['timestamp'] = add_nodes_features(unique_user_id, candidate_pre, 'Name', 'Date Added',
                                                     'nameID') / 1000
    unique_user_id['salary'] = add_nodes_features(unique_user_id, candidate_pre, 'Name', 'salary_current', 'nameID')
    # unique_user_id['payrate'] = add_nodes_features(unique_user_id, candidate_pre, 'Name',
    #                                                'daily_rate', 'nameID')
    # unique_user_id['latitude'] = add_nodes_features(unique_user_id, candidate_pre, 'Name', 'latitude', 'nameID')
    # unique_user_id['longitude'] = add_nodes_features(unique_user_id, candidate_pre, 'Name', 'longitude', 'nameID')

    # Create a mapping from unique Candidatures indices
    # This is a new way to treat the job candidates relationship
    unique_candidature_id = job_candidates['unique_id']
    unique_candidature_id = pd.DataFrame(data={
        'candidatureID': unique_candidature_id,
        'mappedID': pd.RangeIndex(len(unique_candidature_id)),
        'timestamp': job_candidates['Date Added_caller'].apply(lambda x: time.mktime(x.timetuple()))
        # convert to timestamp
    })

    # Create a mapping from unique jobs indices (get unique jobs, not all from job_candidates)
    unique_job_id = job_pre['title'].unique()
    unique_job_id = pd.DataFrame(data={
        'jobID': unique_job_id,
        'mappedID': pd.RangeIndex(len(unique_job_id)),
    })

    # add node data for : timestamp, salary, payrate, latitude, longitude
    unique_job_id['timestamp'] = add_nodes_features(unique_job_id, job_pre, 'title', 'dateAdded', 'jobID')
    unique_job_id['salary'] = add_nodes_features(unique_job_id, job_pre, 'title', 'salary', 'jobID')
    # unique_job_id['payrate'] = add_nodes_features(unique_job_id, job_pre, 'title', 'payRate', 'jobID')
    # unique_job_id['latitude'] = add_nodes_features(unique_job_id, job_pre, 'title', 'latitude', 'jobID')
    # unique_job_id['longitude'] = add_nodes_features(unique_job_id, job_pre, 'title', 'longitude', 'jobID')

    # Create a mapping from unique skills indices to range [0, num_user_nodes):
    unique_skill_id = np.unique(np.append(candidate_skills['skillUri'], job_skills["skillUri"]))
    unique_skill_id = pd.DataFrame(data={
        'skillID': unique_skill_id,
        'mappedID': pd.RangeIndex(len(unique_skill_id)),
    })

    # Create a mapping from unique contract types indices
    unique_contract_id = np.unique(np.append(candidate_contract['contract'], job_contract["contract"]))
    unique_contract_id = pd.DataFrame(data={
        'contractID': unique_contract_id,
        'mappedID': pd.RangeIndex(len(unique_contract_id)),
    })

    # Create a mapping from unique exp indices
    unique_exp_id = np.unique(np.append(candidate_exp['experience'], job_exp["experience"]))
    unique_exp_id = pd.DataFrame.from_dict(data={
        'expID': unique_exp_id,
        'mappedID': pd.RangeIndex(len(unique_exp_id)),
    })

    # Create a mapping from unique origin indices
    unique_origin_id = candidate_origin['origin'].unique()
    unique_origin_id = pd.DataFrame(data={
        'originID': unique_origin_id,
        'mappedID': pd.RangeIndex(len(unique_origin_id)),
    })

    # Create a mapping from unique salary_cat indices
    unique_salary_id = np.array(list(set(np.append(candidate_salary['salaire_cat'], job_salary['salaire_cat']))))
    unique_salary_id = pd.DataFrame(data={
        'salaryID': unique_salary_id,
        'mappedID': pd.RangeIndex(len(unique_salary_id)),
    })

    # Create a mapping from unique zip indices
    # unique_zip_id = np.array(list(set(np.append(candidate_zip['zip'], job_zip['zip']))))
    # unique_zip_id = pd.DataFrame(data={
    #     'zipID': unique_zip_id,
    #     'mappedID': pd.RangeIndex(len(unique_zip_id)),
    # })

    # Create a mapping from unique category indices (jobs only)
    unique_category_id = job_category['category'].unique()
    unique_category_id = pd.DataFrame(data={
        'categoryID': unique_category_id,
        'mappedID': pd.RangeIndex(len(unique_category_id)),
    })

    # Create a mapping from unique company indices (from owner field)
    unique_company_id = job_company_full['company'].unique()
    unique_company_id = pd.DataFrame(data={
        'companyID': unique_company_id,
        'mappedID': pd.RangeIndex(len(unique_company_id)),
    })

    # Create a mapping from unique concept indices
    unique_concept_id = skill_concept['concept'].unique()
    unique_concept_id = pd.DataFrame(data={
        'conceptID': unique_concept_id,
        'mappedID': pd.RangeIndex(len(unique_concept_id)),
    })

    timestamps = list(unique_user_id['timestamp']) + list(unique_candidature_id["timestamp"]) + list(unique_job_id[
        'timestamp'])
    timestamps = [datetime.fromtimestamp(x) for x in timestamps if not pd.isna(x)]
    year_months = list(set([d.year * 12 + d.month - 1 for d in timestamps]))
    year_months_timestamps = [int(datetime(x // 12, x % 12 + 1, 1).timestamp()) for x in year_months]
    min_year_months = min(year_months)
    min_year_months_ts = min(year_months_timestamps)
    year_months = [d - min_year_months for d in year_months]

    unique_time_id = pd.DataFrame(data={
        'timeID': year_months,
        'mappedID': pd.RangeIndex(len(year_months)),
        "timestamp": year_months_timestamps
    })

    unique_user_id["yearmonth"] = [datetime.fromtimestamp(x) if not pd.isna(x)
                                   else datetime.fromtimestamp(min_year_months_ts)
                                   for x in unique_user_id['timestamp']]
    unique_user_id["yearmonth"] = [d.year * 12 + d.month - 1 - min_year_months
                                   for d in unique_user_id["yearmonth"]]
    unique_job_id["yearmonth"] = [datetime.fromtimestamp(x) if not pd.isna(x)
                                  else datetime.fromtimestamp(min_year_months_ts)
                                  for x in unique_job_id['timestamp']]
    unique_job_id["yearmonth"] = [d.year * 12 + d.month - 1 - min_year_months
                                  for d in unique_job_id["yearmonth"]]
    unique_candidature_id["yearmonth"] = [datetime.fromtimestamp(x) if not pd.isna(x)
                                          else datetime.fromtimestamp(min_year_months_ts)
                                          for x in unique_candidature_id['timestamp']]
    unique_candidature_id["yearmonth"] = [d.year * 12 + d.month - 1 - min_year_months
                                          for d in unique_candidature_id["yearmonth"]]

    return (unique_user_id, unique_skill_id, unique_job_id, unique_candidature_id, unique_contract_id, unique_exp_id,
            unique_origin_id, unique_salary_id, unique_category_id,
            unique_company_id, unique_concept_id, unique_time_id)


def load_edges(
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_exp,
        job_exp,
        candidate_origin,
        candidate_contract,
        job_contract,
        job_category,
        candidate_salary,
        job_salary,
        job_company,
        job_company_full,
        skill_concept,
        unique_user_id,
        unique_skill_id,
        unique_job_id,
        unique_candidature_id,
        unique_contract_id,
        unique_exp_id,
        unique_origin_id,
        unique_salary_id,
        unique_category_id,
        unique_company_id,
        unique_concept_id,
        unique_time_id
):
    # candidature to job
    edge_index_candidature_to_job = get_edge_index('unique_id', 'candidatureID', 'Job', 'jobID', job_candidates,
                                                   unique_candidature_id, unique_job_id)
    # candidature to candidate
    edge_index_user_to_candidature = get_edge_index('name', 'nameID', 'unique_id', 'candidatureID', job_candidates,
                                                    unique_user_id, unique_candidature_id)

    # candidate to job
    edge_index_user_to_job, edge_index_user_to_job_time = get_edge_index('name', 'nameID', 'Job', 'jobID',
                                                                         job_candidates, unique_user_id, unique_job_id,
                                                                         with_timestamp=True, timestamp='timestamp')

    # Candidate to time
    edge_index_candidate_to_time = get_edge_index('nameID', 'nameID',
                                                  "yearmonth", "timeID",
                                                  unique_user_id, unique_user_id, unique_time_id, from_pre=True)
    # Candidature to time
    edge_index_candidature_to_time = get_edge_index('candidatureID', 'candidatureID',
                                                    "yearmonth", "timeID",
                                                    unique_candidature_id, unique_candidature_id, unique_time_id, from_pre=True)

    # Job to time
    edge_index_job_to_time = get_edge_index('jobID', 'jobID',
                                            "yearmonth", "timeID",
                                            unique_job_id, unique_job_id, unique_time_id, from_pre=True)

    # skill to candidate
    edge_index_user_to_skill = get_edge_index('name', 'nameID', 'skillUri', 'skillID', candidate_skills, unique_user_id,
                                              unique_skill_id)

    # skill to job
    edge_index_job_to_skill = get_edge_index('Job', 'jobID', 'skillUri', 'skillID', job_skills, unique_job_id,
                                             unique_skill_id)

    # candidate to contract
    edge_index_user_to_contract = get_edge_index('name', 'nameID', 'contract', 'contractID', candidate_contract,
                                                 unique_user_id, unique_contract_id, from_pre=True)

    # job to contract
    edge_index_job_to_contract = get_edge_index('Job', 'jobID', 'contract', 'contractID', job_contract, unique_job_id,
                                                unique_contract_id, from_pre=True)
    # candidate to exp
    edge_index_user_to_exp = get_edge_index('name', 'nameID', 'experience', 'expID', candidate_exp, unique_user_id,
                                            unique_exp_id, from_pre=True)

    # job to exp
    edge_index_job_to_exp = get_edge_index('Job', 'jobID', 'experience', 'expID', job_exp, unique_job_id, unique_exp_id,
                                           from_pre=True)

    # candidate to origin
    edge_index_user_to_origin = get_edge_index('name', 'nameID', 'origin', 'originID', candidate_origin, unique_user_id,
                                               unique_origin_id, from_pre=True)

    # candidate to zip
    # edge_index_user_to_zip = get_edge_index('name', 'nameID', 'zip', 'zipID', candidate_zip, unique_user_id,
    #                                         unique_zip_id, from_pre=True)

    # job to zip
    # edge_index_job_to_zip = get_edge_index('Job', 'jobID', 'zip', 'zipID', job_zip, unique_job_id, unique_zip_id,
                                        #    from_pre=True)

    # candidate to salary
    edge_index_user_to_salary = get_edge_index('name', 'nameID', 'salaire_cat', 'salaryID', candidate_salary,
                                               unique_user_id, unique_salary_id, from_pre=True)

    # job to salary
    edge_index_job_to_salary = get_edge_index('Job', 'jobID', 'salaire_cat', 'salaryID', job_salary, unique_job_id,
                                              unique_salary_id, from_pre=True)

    # candidate to category
    # edge_index_user_to_category = get_edge_index('name', 'nameID', 'category', 'categoryID', candidate_category,
    #                                              unique_user_id, unique_category_id, from_pre=True)

    # job to category
    edge_index_job_to_category = get_edge_index('Job', 'jobID', 'category', 'categoryID', job_category, unique_job_id,
                                                unique_category_id, from_pre=True)

    # job to company (using owner field)
    edge_index_job_to_company = get_edge_index('Job', 'jobID', 'company', 'companyID', job_company_full, unique_job_id,
                                               unique_company_id, from_pre=True)

    # skill to concept
    edge_index_skill_to_concept = get_edge_index('skillUri', 'skillID', 'concept', 'conceptID', skill_concept,
                                                 unique_skill_id, unique_concept_id, from_pre=True)

    return (edge_index_user_to_job, edge_index_user_to_skill, edge_index_job_to_skill, edge_index_user_to_contract, edge_index_job_to_contract,
            edge_index_user_to_exp, edge_index_job_to_exp, edge_index_user_to_origin, edge_index_user_to_salary,
            edge_index_job_to_salary, edge_index_job_to_category,
            edge_index_job_to_company,
            edge_index_skill_to_concept, edge_index_user_to_job_time, edge_index_candidature_to_job,
            edge_index_user_to_candidature, edge_index_candidate_to_time, edge_index_candidature_to_time,
            edge_index_job_to_time)


def load_features(unique_job_id, unique_user_id, candidate, reload=False):
    # if reload:
    #     model_sent = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # fastest

    #     sentences = unique_job_id["jobID"].fillna("")
    #     job_features = model_sent.encode(sentences)

    #     candidate_cv = pd.DataFrame()
    #     candidate_cv['name'] = unique_user_id['nameID']
    #     candidate_cv = candidate_cv.merge(candidate, on='name', how='left')
    #     sentences = candidate_cv['description'].fillna("")
    #     candidate_features = model_sent.encode(sentences)

    #     # export features
    #     np.save('save/job_features.npy', job_features)
    #     np.save('save/candidate_features.npy', candidate_features)

    # load features
    job_features = np.load(DEFAULT_SAVE_DIR / 'job_features.npy')
    candidate_features = np.load(DEFAULT_SAVE_DIR / 'candidate_features.npy')

    # if job_features.shape[0] != len(unique_job_id) or candidate_features.shape[0] != len(unique_user_id):
    #     print("We need to recompute features...")
    #     return load_features(unique_job_id, unique_user_id, candidate, reload=True)
    return job_features, candidate_features


"""
    abl_list : list of 10 elements, each element is 1 or 0, 1 means that the feature is used, 0 means that the feature is not used
    candidature_node : boolean, if True, add the candidature node to the graph
    ts_nodes : boolean, if True, add timestamp to each node with temporal information
    ts_nodes_all : boolean, if True, add timestamp to all nodes initialized at 1 january 1970
"""


def build_graph(path=DEFAULT_DATA_DIR,
                abl_list=None,
                candidature_node=False,
                ts_nodes=False,
                ts_nodes_all=False,
                ts_attr=True,
                percentile_min=0,
                percentile_max=100,
                error_analysis=None):
    if abl_list is None:
        abl_list = [1, 1, 1, 1, 1, 1, 1, 1]  # skill, contract, origin, exp, salary, category, company, concept, time
    # Load data
    (candidate, candidate_skills, job_candidates, job_skills, candidate_pre, job,
     job_pre, skill, hierarchy, candidate_exp, job_exp, candidate_origin, candidate_contract, job_contract,
     job_category, candidate_salary, job_salary, job_company, job_company_full, skill_concept) = load_data(path)
    print("Data loaded")

    # load nodes
    (unique_user_id, unique_skill_id, unique_job_id, unique_candidature_id, unique_contract_id, unique_exp_id,
     unique_origin_id, unique_salary_id, unique_category_id,
     unique_company_id, unique_concept_id, unique_time_id) = load_nodes(
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_pre,
        candidate_exp,
        job_exp,
        candidate_origin,
        candidate_contract,
        job_contract,
        job_category,
        candidate_salary,
        job_salary,
        job_company,
        job_company_full,
        skill_concept,
        job_pre
    )
    print("Nodes loaded")

    # load edges
    (edge_index_user_to_job, edge_index_user_to_skill, edge_index_job_to_skill, edge_index_user_to_contract, edge_index_job_to_contract,
     edge_index_user_to_exp, edge_index_job_to_exp, edge_index_user_to_origin, edge_index_user_to_salary,
     edge_index_job_to_salary, edge_index_job_to_category,
     edge_index_job_to_company, edge_index_skill_to_concept,
     edge_index_user_to_job_time, edge_index_candidature_to_job, edge_index_user_to_candidature,
     edge_index_candidate_to_time, edge_index_candidature_to_time, edge_index_job_to_time) = load_edges(
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_exp,
        job_exp,
        candidate_origin,
        candidate_contract,
        job_contract,
        job_category,
        candidate_salary,
        job_salary,
        job_company,
        job_company_full,
        skill_concept,
        unique_user_id,
        unique_skill_id,
        unique_job_id,
        unique_candidature_id,
        unique_contract_id,
        unique_exp_id,
        unique_origin_id,
        unique_salary_id,
        unique_category_id,
        unique_company_id,
        unique_concept_id,
        unique_time_id
    )

    if error_analysis is not None:
        error_analysis_dir = Path(error_analysis)
        error_analysis_dir.mkdir(parents=True, exist_ok=True)
        unique_user_id.to_csv(error_analysis_dir / "user_id.csv", index=False)
        unique_job_id.to_csv(error_analysis_dir / "job_id.csv", index=False)
        print("Saved preprocessed data for error analysis.")

    print("Edges loaded")

    # load features
    job_features, candidate_features = load_features(unique_job_id, unique_user_id, candidate, True)
    print("Features loaded")

    # build graph
    data = HeteroData()  # Save node indices:
    data["candidate"].node_id = torch.arange(len(unique_user_id))

    data["candidate"].x = torch.Tensor(candidate_features)
    if ts_nodes:
        data["candidate"].timestamp = torch.Tensor(unique_user_id['timestamp']).long()

    data["job"].node_id = torch.arange(len(unique_job_id))  # Add the node features and edge indices:

    data["job"].x = torch.Tensor(job_features)
    if ts_nodes:
        data["job"].timestamp = torch.Tensor(unique_job_id['timestamp']).long()

    if candidature_node:
        data["candidature"].node_id = torch.arange(len(unique_candidature_id))
        if ts_nodes:
            data["candidature"].timestamp = torch.Tensor(unique_candidature_id['timestamp']).long()

    # From now on node don't have timestamp, but we will add a standard timestamp to each node
    # Initializing all nodes timestamps to 1 january 1970

    if abl_list[0] == 1:
        data["skill"].node_id = torch.arange(len(unique_skill_id))  # Add the node features and edge indices:
        if ts_nodes_all:
            data["skill"].timestamp = torch.Tensor(np.zeros(len(unique_skill_id))).long()

    if abl_list[1] == 1:
        data["contract"].node_id = torch.arange(len(unique_contract_id))  # Add the node features and edge indices:
        if ts_nodes_all:
            data["contract"].timestamp = torch.Tensor(np.zeros(len(unique_contract_id))).long()

    if abl_list[2] == 1:
        data["origin"].node_id = torch.arange(len(unique_origin_id))
        if ts_nodes_all:
            data["origin"].timestamp = torch.Tensor(np.zeros(len(unique_origin_id))).long()
    if abl_list[3] == 1:
        data["experience"].node_id = torch.arange(len(unique_exp_id))
        if ts_nodes_all:
            data["experience"].timestamp = torch.Tensor(np.zeros(len(unique_exp_id))).long()

    if abl_list[4] == 1:
        data["salary"].node_id = torch.arange(len(unique_salary_id))
        if ts_nodes_all:
            data["salary"].timestamp = torch.Tensor(np.zeros(len(unique_salary_id))).long()

    if abl_list[5] == 1:
        data["category"].node_id = torch.arange(len(unique_category_id))
        if ts_nodes_all:
            data["category"].timestamp = torch.Tensor(np.zeros(len(unique_category_id))).long()

    if abl_list[6] == 1:
        data["company"].node_id = torch.arange(len(unique_company_id))
        if ts_nodes_all:
            data["company"].timestamp = torch.Tensor(np.zeros(len(unique_company_id))).long()

    if abl_list[7] == 1:
        data["concept"].node_id = torch.arange(len(unique_concept_id))
        if ts_nodes_all:
            data["concept"].timestamp = torch.Tensor(np.zeros(len(unique_concept_id))).long()

    if ts_attr:
        data["time"].node_id = torch.arange(len(unique_time_id))
        if ts_nodes:
            data["time"].timestamp = torch.Tensor(unique_time_id["timestamp"]).long()
        data["time"].x = torch.Tensor(np.array([[x] for x in unique_time_id["timeID"]]))

    print("Number of edges from candidature to job", len(edge_index_candidature_to_job[0]))
    min_index = round(percentile_min / 100 * len(edge_index_candidature_to_job[0]))
    max_index = round(percentile_max / 100 * len(edge_index_candidature_to_job[0]))
    print('min_index : ', min_index)
    if candidature_node:
        # About candidature
        if percentile_max - percentile_min != 100:
            edge_index_tmp_0 = edge_index_candidature_to_job[0].numpy()[min_index:max_index]
            edge_index_tmp_1 = edge_index_candidature_to_job[1].numpy()[min_index:max_index]
            edge_index_candidature_to_job = torch.Tensor([edge_index_tmp_0, edge_index_tmp_1]).long()

        data["candidature", "has_application", "job"].edge_index = edge_index_candidature_to_job
        print('edge_index_candidature_to_job : ', edge_index_candidature_to_job)
        print('edge_index_user_to_candidature : ',edge_index_user_to_candidature)
        print(edge_index_user_to_candidature.size())
        print("edge_index_user_to_skill",edge_index_user_to_skill)
        print(edge_index_user_to_skill.size())
        data["candidate", "applied_with", "candidature"].edge_index = edge_index_user_to_candidature
        if ts_attr:
            data["candidature", "has_time", "time"].edge_index = edge_index_candidature_to_time
    else:
        # about jobs
        data["candidate", "is_shortlist", "job"].edge_index = edge_index_user_to_job
        # data["candidate", "is_shortlist", "job"].timestamp = torch.Tensor(edge_index_user_to_job_time).long()

    # job to candidate
    # data["job", "shortlist", "candidate"].edge_index = edge_index_user_to_job

    # about skill
    if abl_list[0] == 1:
        data["candidate", "has", "skill"].edge_index = edge_index_user_to_skill
        data["job", "has", "skill"].edge_index = edge_index_job_to_skill

    # about contract
    if abl_list[1] == 1:
        data["candidate", "work_on", "contract"].edge_index = edge_index_user_to_contract
        data["job", "has_type", "contract"].edge_index = edge_index_job_to_contract

    # about origin
    if abl_list[2] == 1:
        data["candidate", "was_found_on", "origin"].edge_index = edge_index_user_to_origin

    # about exp
    if abl_list[3] == 1:
        data["candidate", "has_gain", "experience"].edge_index = edge_index_user_to_exp
        data["job", "requires", "experience"].edge_index = edge_index_job_to_exp

    # about salary
    if abl_list[4] == 1:
        data["candidate", "is_worth", "salary"].edge_index = edge_index_user_to_salary
        data["job", "is_worth", "salary"].edge_index = edge_index_job_to_salary

    # about category (jobs only)
    if abl_list[5] == 1:
        data["job", "is_in", "category"].edge_index = edge_index_job_to_category

    # about company
    if abl_list[6] == 1:
        data["job", "is_for", "company"].edge_index = edge_index_job_to_company

    # about concept
    if abl_list[7] == 1:
        data["skill", "is_attached_to", "concept"].edge_index = edge_index_skill_to_concept

    if ts_attr:
        data["candidate", "has_time", "time"].edge_index = edge_index_candidate_to_time
        data["job", "has_time", "time"].edge_index = edge_index_job_to_time

    # We also need to make sure to add the reverse edges from job to candidate
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)

    return data


if __name__ == '__main__':
    graph = build_graph(DEFAULT_DATA_DIR, candidature_node=True, ts_nodes=True, ts_nodes_all=True)
