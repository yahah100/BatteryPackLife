# Use command: python download.py [dataset name]
# Example: If you want to download MATR dataset, please use command: python download.py MATR
# Where to find the datasets: Datasets may be downloaded in the ./dataset file

import os
import sys
import shutil
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

def download_file_from_github(dataset_name):
    url_map={
        'CALCE': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/CALCE.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144149Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=062e62883561be8d2efd0c2beacdf1f2d6a1f8ce020e854ea0a0bd18222fee60456dcf86ab853d2acea8277d797766a5acd479162cda94361abb37fa9282e700ab2681851d0b100d92b30a2eda22a43ee84f71b0cba507b37af2a8eb89290331f3361b9a0290f96cce128e215d8ee3173669426d7e732be6f4c38372ca83f29618c5edc6a228b9ec315b944e301c61c0e8cf361729b9361b886701aab686f030a1841c1989d37fe7d4a249f922ed91301f24d79f0e58a45e0a4bacb9d40db161bc5ec8dcdf2758fd495af2c15f5660dc68b77d4a19fe5c344250b2446693f82c1c2c3bb323ddb5daf4bf921968882637312d71527acf1f609b4fca1e3a353bd1',
        'HNEI': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/HNEI.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144217Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=13822e9e2dbcbf3f95c1b404dc673fc2ee57fcd28485dfd48de61decd46d309f1ce5a00fce735d498d7ed12cc380e705a12862389d158f7c93a2f7c22a7f6fc7b913c63093d7365f5a205b45b045715908b6b1e6e522267e94e469b69393daaf9ad030e688a1fbd5deafb8fdf9aed3e6370ce25ad0b3ab7e618aa5dceff3c7e8676bca5f3f56fd02fc91f71c08666e1414fab1477699c90fabac69e3d7a9d6aab0a508da685db9d2049e7d2c6781e35c6c4dd493455ca9682d2fd5ec73868b5c53d00ea20519082b9a33c661d08e7f5b3701c58042539213276ae70d5f2290f2611bd3421684873d97d0e6d4c602f254ff506d77638b419ee488979e74e47e8b',
        'HUST': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/HUST.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144250Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=94e6818daf73cfe19a45179f625b951179c9d787048a4191fbb7bc0484d3619e0cd0b1d8413ac1a70247fc2f189f0e19022c320de0ff7113b1d3c123c4f5d04154d24e63c408bea6119071eb8a23e0b1b5004bb07454c98015d8ffe46697c4049c3f43ba796970b7f278d384d4cb47d187b0bbbafb0f127eab2068f0382c7330335ca3219f477769bf57fe39db14e8f9c7ac23d49aced5ebe017d71054b3fdc0d526b51c8e7cd817bb03eef6a4a40b14525cb77aa5cbc93d829dbe3e119eeb58f5b3201a30390af3a94959bfd6dc8b5a28d280494acec64d602e6fac07bebe61a14a947e3a93f0ad6e91d6ed0c1c624efda931e7428157b0b37692654c7034db',
        'MATR': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/MATR.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144344Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5f682c3f5481d1479efb93d5336351bb32793f3dc5c67cfb7fd23d1f6f689aa773c51c99ab0e06505ecb9fc7ad3282f70fffc3e7a30093b1b50df62a8254522f84449c7fdc957c27e2be6434ad68758d21a3ac2bde6f014a329edf25aa4b091bc829e606a5ae60bce45f91a17fd216fc39aa92cc4383402e3ac340d90e1cdc545bd0697b96f99970d9d4222e7a822ff5a27ce46f703a63a64ddc2a8e029ebfe5d9ebf4962d7eb602caedd2a5828ef2e7fb32b2e3129c4d7fff6bed488e91d505ddba57a1ab5e24f30a7cd22533a4c9d58ed960f153673785c403cd9d30869dce4ee5b887b1aaafa1b032faa33088b86659c05cc6b60e7b6b8f3bc89132e51e69',
        'RWTH': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/RWTH.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144529Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=146249b923c7d4913ebdf1732e3e074860c6900e7871b3f6e9ba0a3ff8566c0a1d3cbe16db61409123c17c8a4d66df20fd8b7366dc617eca124c973fde0558752a57cc543e17cc9a5b7aa77f399b390be155e9014ccbb32c2377f29246122667b88660309c7ec9c3bad3f0f714bd557333d13d3535564baf48d1926610673c6bdcbf9b378c12de5e09a038520edc2fb65a67c04a1f85a3bdb270bc643407a64592195333cf8e4068d9f356b1485c3e3e5af50488e9e902bb1551e2ff82119cd6c296afd7b1028e1009e3dc5b2ffb81394c3c111453ea8be89865589b4f40b2d53fe76ee80bc3d50c8a359d8316381f0c22b4a055ba346b3bc72eaf5c4773ec72',
        'SNL': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/SNL.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144542Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=06fc28590dfd0aa7234b0cbe54c588c1d60673d7a7590d566a6f2e0a0c3d5f05275c3c1fbdcf61893fe37bb8face515a884eb101b13ecf903dc377811ab7045afa3c3abebf6537f55efb1b4263827bfee9b995263a3b0ca67ea234514a7b091c0865cbfbfd79a4966c1d683d1c4dd55e10fdab84396430a2b226308bab240c92093cebe87f2bb50e7d0da52b27782c320f5a9bc5d62fd4258f4846d0cda865f1f7e24938044dd6fe0b6b9dd4456f348ac6cc9ebcb672bb06f679d3eb9ef7b757e8dd61a0b89be7d7475577ce9a4e7c1eca3dbc25d3a94e74e9dfc9b6fb21c273b9e1075874215998f7261d77aa7e45eeb8799917f6b16efb0c1289ef32c67821',
        'MICH': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/MICH.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144405Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5066be86edeb5689dee1fe8a744a9af6e4981172513e4fd73479b26cdae6aec0bd678c946f83c17a8737b910fcb89a05ea9b3fd750305e305adaf9528b1e0242a27a2fded5f4ab423493735a07a270b97ac4924693cdc243dc372fc86ed51f060bd1b5e32d1c01947a75d5759de12a72c5204e4cf798dfc4fd07ab4ce44ebfd07053cfc06d1657b1cca7f5a98a7c2df91c6bc31cdb017315e6cd717a53c25470def1641d035df667c4bf0052b6a270c7b9abb4e758ec8e232eea653c1ba7592e17b190fe4bb4b8d09435281f6a0fd916a857f4a3d1563c952ac503ce4226108f3fd9bd0d048ab4c95858619bba0af66c3b7235f4d9e97dc459bb2899a8f10e49',
        'MICH_EXP': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/MICH_EXP.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144452Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9ca8448a2eb262d334bf95e70bc7ef33898dbc87668a76ea3eb02480744405753ca609968467467b4494ef7b77a317472b2845fb1a74aa55cb004c640371ac731a10ce22d83f902aca408c1e5577198ff69b6e0855a2f2e902633f49b49680970966d7b85ee243a232289f41a6a940ca21665e3c92a0b73a98d7189c594d8740fcf5e8c4fb8fc50d33578c87c300585e12bc8052e2ba972878a9fe4362e95dd61e704d1f49a41709a8e5f5959c3c066cdf4a3213b68a6f2912e76c209781f504b25046a87269c2ba26cf08912519b2628b30b1a2af8ac9bc4953c571dcf7615202d5c50d1b27fd503ae80a5156f8d98108abf18518ad4f264dbb7af6e5031cf5',
        'Tongji': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/Tongji.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144623Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5e9516e9bf02d5ddc70b54746ee29286cabae6ac8188c47823e2946139a9d4501640024a7d235ef233646093e9f345fd2a678bdb67d788961982e55aa3d65ab46af5b4a16403955f104f927da94a3c43c00b80c182dece52bb00f7d0781ba9950a0f9cadaab038fbc2d80a5c81dde3d96a11bae8a243cd4f8de8b21f0f69e021882e8d9cc67745e86f04e1f789def206374168839c9186638b2e53d185bf0f52558c71965dee2b2e9abe7aca45bbe5bd9d03e4cce8c9518e27817e982161feef3a8710c22d1c6735dcc09f634ff765c52e6ed24efb3e61479d5da0e6b56141fd0e1dab378d12d1a5b75f831e8797aabba45581c96b1a7550fbc890a2ba222ff5',
        'Stanford': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/Stanford.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144601Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=659ba98c5e1f2acff7bb0c90accde9228c68f2d39edbb5ffb182e64ea09342e0cd311c29b9a710d17a278d1aab41a0968a71e918c40aa512f83600d03a14a8d991a941cad51a9efdd43c0ac731aecb48ae91116f52413b5fb6b9e4e7ed9e952921f6fc87c9d61cc0baa3f825bab56b0707b9fe1c42491b8095f5aff0c95bb4b4fada10f32107019d0b2ae7860213db606f30a5ee8e78f1e5cdd0bd478b8ff0d3fc4d1e80490faeb6d22933c9f456d841a41a2857e088e0443306322f28be872647b80a4f8f2208572ba398aac25b71fcd50008a9d7f567476a1a89d5d6740e4205d62e7128feebda16239d66e21c8fa44199161b18284fcb453a6802f1d0cd38',
        'ISU_ILCC': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/ISU_ILCC.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144312Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3a0c263daa995a22d09fc4e99e62b86700ca2ed4d0074bcefe2890f60ff0526ee4cfbc64d6edaec975d1fa16911ee5add8d19b67a40a41eaf7b7444211096fe2299a0fb0a12bcd0a8271718f6524119c1590651fa233f5870478a31ff564163e3d4bb484df737b6a92e9b5ccb67553af99bce63e1f41efa54868db86e96f2de0fcfda30ebdcac9fd59e65b892a4ecf5593e91acd244b5888302e58ad097a4621bdc9395f846a1139b0e887c48de595dda4c1ca4515ba2798026a04cbd8c59ea9c17267b9b8ed1d6b3a14c2f1eaf064fbb709abea7734fbcfa5765b1394a46dd345657e5b96c97c7a25c659868997c75a1912ff7e5ddc36add552678e43dc4776',
        'XJTU': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9768320/upload/XJTU.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241031%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241031T030817Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=25a656d592f394385d9a34db84d901586a9b208318dce7eb593c136fa73da373ee44cd9d7e1434f105112eaf0060b57b191a0aeec6b280bde706a59e7eb2fd6cd8df1916fe95889b7119a3f337fe0b66e1d0b9be38684648f3a4f5d7d1be697c36d8ce0b8f2486ea142f4bac1888bfcf2c6f3f711cca5db8e2908aabacff04016678940c211b9c364359d9d7d2d0c58144188316bb3bf97f6a78b3aa280fc7a57d9bc50be6a37bd7203c87aa7c216021e488338284d9d0d22dbbd0895d0b791de8f1b74fda9d579f0b243ad219ac9b2ca0d34e332272b32194c245ef6fd595b3a4fd742bb60ff538015f2708a5155c0d1fc763bf900eb4777b556b282ea491cd',
        'ZN-coin': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9751440/upload/ZNcoin.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241030%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241030T144706Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7d1c96e04002aaa8b9e3224d2715636db121685b40f3cef45f6028fdc5bbd07379fc3d757122a0c1432bec7b5b2dc04ee63640a87cdda3a4df8dfb2181d1c5397db7d447317b1a8ac108e69e85cb025a1b9faab8d57eea1d61fd5bcd69d2513edf925d4f1150b0dd31878c0b457c689024ace1472735d336b10e4c50631775f752d62328b50ab91e8df27bc110467f3f7a23cd1325753c123a744ec5b08562ea7095a1864d8e0309f438d011f5ea7d1a3a8bb9acbf5e9f049c69e4fd4df71dab036494ddfa4b0f45000dbf54c3243bfc04d566795aaa085c06dc8fbf0c26146147786ec529b6550431f23740bbb0f3b2bb107245c29d90e3125b394b7709735d',
        'UL-PUR': 'https://storage.googleapis.com/kaggle-data-sets/5677689/9768320/upload/UL_PUR.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241104%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241104T124625Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=708e78aad159c14654075ad5ae7392c8b2cc1949ea714082fc66331650f7f588d3f5e039fd5abbe95fc4269429abbfa2d61548caeaed10bc2e8733f583f265fd1a27f4374644f9b8410428fdeeb62425f83ecc7bb2e469e2e646120beac4a186c12d5a9790372a4dff10fecdaa3e65855201816950faa9e1c3cc48832ff7c7fbbe383313613ed1b8bdd0cbc8c4d947398f1eff1c2c882b6730efc82a258ea16c00723f764a0cb500b9f66743c009a3e8914b2640f40e1147b3fb07114ede5f4c6157fbdeab5b42a76a5cfa36453f6aca93b4a2f7c6eb2db9062da423f66eada637e64f27528f853184c63d3816af95f912e871202bbc800169283196a829cc9e',
    }

    url = url_map[dataset_name]
    local_file = f'./dataset/{dataset_name}.zip'

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
        with open(local_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        # progress_bar.close()
        print(f"Data has been downloaded to {local_file}")

        # Special operation for Tongji dataset
        if dataset_name == 'Tongji':
            path = './dataset'
            raw_file = Path(path) / 'Tongji.zip'
            # Unzip the raw file
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                tqdm.write('Unzipping Tongji datasets and rename the files.')
                pbar = zip_ref.namelist()
                for file in tqdm(pbar):
                    zip_ref.extract(file, path)

            files_path = os.listdir(path + '/Tongji/')
            files = [i for i in files_path if i.endswith('.pkl')]
            for file in tqdm(files):
                os.rename(path + '/Tongji/' + file, path + '/Tongji/' + file.split('--')[0] + '-#' + file.split('--')[1])

            os.remove(raw_file)
            print('Delete the old version of Tongji dataset,')

            zip_filename = Path(path) / 'Tongji.zip'
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                tqdm.write('Compress the Tongji dataset')
                for file in tqdm(os.listdir(Path(path) / 'Tongji')):
                    file_path = Path(path) / 'Tongji' / file
                    if file_path.is_file():
                        zipf.write(file_path, arcname=file)

            shutil.rmtree(raw_file.parent / 'Tongji')
            print('Complete rename operations.')

    else:
        print(f"Download failed: {response.status_code}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    download_file_from_github(dataset_name)
