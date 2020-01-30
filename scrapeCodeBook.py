import requests
from bs4 import BeautifulSoup as BS
import re
import json

hanes_2013_files = ['ALB_CR_H', 'ALDS_H', 'ALD_H', 'AMDGDS_H', 'AMDGYD_H', 'APOB_H',
                    'BFRPOL_H', 'BIOPRO_H', 'CBC_H', 'CHLMDA_H', 'COT_H', 'CUSEZN_H',
                    'DEET_H', 'DEMO_H', 'EPHPP_H', 'ETHOXS_H', 'ETHOX_H', 'FASTQX_H', 'FLDEP_H',
                    'FLDEW_H', 'FOLATE_H', 'FOLFMS_H', 'GHB_H', 'GLU_H', 'HCAAS_H',
                    'HCAA_H', 'HDL_H', 'HEPA_H', 'HEPBD_H', 'HEPB_S_H', 'HEPC_H',
                    'HEPE_H', 'HIV_H', 'HPVP_H', 'HPVSWR_H', 'HSV_H', 'IHGEM_H',
                    'INS_H', 'MMA_H', 'OGTT_H', 'ORHPV_H', 'PAH_H', 'PBCD_H',
                    'PCBPOL_H', 'PERNTS_H', 'PERNT_H', 'PFAS_H', 'PHTHTE_H', 'POOLTF_H',
                    'PSTPOL_H', 'SSFLRT_H', 'SSHEPC_H', 'SSPFAC_H', 'SSPFAS_H', 'SSPFSU_H',
                    'SSPHTE_H', 'SSTOCA_H', 'SSTOXO_H', 'TCHOL_H', 'TGEMA_H', 'TRICH_H',
                    'TRIGLY_H', 'TSNA_H', 'TST_H', 'UASS_H', 'UAS_H', 'UCFLOW_H',
                    'UCOTS_H', 'UCOT_H', 'UCPREG_H', 'UHG_H', 'UIO_H', 'UMS_H', 'UM_H',
                    'UTASS_H', 'UTAS_H', 'UVOCS_H', 'UVOC_H', 'VID_H', 'VITB12_H',
                    'VNAS_H', 'VNA_H', 'VOCWBS_H', 'VOCWB_H', ]

hanes_2013_questionnaire_files = ['ACQ_H', 'ALQ_H', 'BPQ_H', 'CBQ_H', 'CDQ_H', 'CFQ_H',
                                  'CKQ_H', 'CSQ_H', 'DBQ_H', 'DEQ_H', 'DIQ_H', 'DLQ_H',
                                  'DPQ_H', 'DUQ_H', 'ECQ_H', 'FSQ_H', 'HEQ_H', 'HIQ_H',
                                  'HOQ_H', 'HSQ_H', 'HUQ_H', 'IMQ_H', 'INQ_H', 'KIQ_U_H',
                                  'MCQ_H', 'OCQ_H', 'OHQ_H', 'OSQ_H', 'PAQ_H', 'PFQ_H',
                                  'PUQMEC_H', 'RHQ_H', 'RXQASA_H', 'RXQ_RX_H',
                                  'SLQ_H',
                                  'SMQFAM_H', 'SMQRTU_H', 'SMQSHS_H', 'SMQ_H', 'SXQ_H', 'VTQ_H',
                                  'WHQMEC_H', 'WHQ_H',
                                  ]

def get_soup_page(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    page = requests.get(url, headers=headers, timeout=5)
    soup = BS(page.content, 'html.parser').html
    return soup


def get_code_book(soup, code_book, fle, fle_code_book):
    key_pat = re.compile('([\w]+)')
    # \w - any word character  \W - any non-alpha i.e whitespace & punctuation
    val_pat = re.compile('\-\s+([\w\W]+)')

    elem_list = soup.find(id='CodebookLinks')

    elems = elem_list.find_all('li')
    for e in elems:
        dict_text = e.find('a').string
        # print(f"Dict text: {dict_text}")
        key_text = key_pat.search(dict_text)
        ky = key_text.group(1).lower()
        val_text = val_pat.search(dict_text)
        vl = val_text.group(1)
        print(f"Dict text: {dict_text}  key: {ky} val: {vl}")
        code_book[ky] = vl

        if ky in fle_code_book:
            fle_code_book[ky].append(fle)
        else:
            fle_code_book[ky] = [fle]
    return code_book, fle_code_book


def get_all_code_books(files, out_file):
    url_prefix = 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/'
    url_postfix = '.htm#Codebook'

    # Initialize with mortality codes
    code_book = {
        'eligstat': 'eligibility status for mortality follow-up',
        'mortstat': 'final mortality status',
        'ucod_leading': 'underlying leading cause of death: recode',
        'diabetes': 'diabetes flag from multiple cause of death (mcod)',
        'hyperten': 'hypertension flag from multiple cause of death (mcod)',
        'permth_int': 'number of person months of follow-up from nhanes interview date',
        'permth_exm': 'number of person months of follow-up from nhanes mobile examination center (mec) date',
    }

    fle_code_book = {}

    for fle in files:
        print(f"File is: {fle}")
        url = url_prefix + fle + url_postfix
        soup = get_soup_page(url)
        code_book, fle_code_book = get_code_book(soup, code_book, fle, fle_code_book)

    print(code_book)
    with open(out_file, 'w') as fp:
        json.dump(code_book, fp, indent=4, separators=(',', ': '))

    # Remove singular values from the file_code_book
    keys_to_del = []
    for k in fle_code_book:
        if len(fle_code_book[k]) == 1:
            keys_to_del.append(k)
    for k in keys_to_del:
        del (fle_code_book[k])
    with open('file_code_book.json', 'w') as fp:
        json.dump(fle_code_book, fp, indent=4, separators=(',', ': '))


get_all_code_books(hanes_2013_questionnaire_files, "qcode_book.json")
