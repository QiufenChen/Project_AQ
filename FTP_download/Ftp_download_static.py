"""
purpose: Complete the download class!
author: Qiufen.Chen
creation_date: 29/11/2019
modification_date: 05/02/2020
"""

import wget
import os

class Ftp:
    # ----------------------------------------------------------------------------------
    def mkdir(self, path):
        """Created download floder!"""
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + " is created successfully!")
            return True
        else:
            return False

    # ----------------------------------------------------------------------------------
    @staticmethod
    def static_download(url):
        """Gain files_list from the website!"""
        filename = wget.download(url)
        return filename    #

    # ----------------------------------------------------------------------------------
    @staticmethod
    def static_readTxt(filename, url):
        """According to the list of files to achieve batch download!"""
        with open(filename, "r") as f:
            for line in f.readlines():
                columns = line.split()
                print(columns[-1])
                full_url = url + str(columns[-1])
                wget.download(full_url)
                print(columns[-1] + " is downloaded successfully!")

    # ---------------------------------------------------------------------------------
    def main(self):
        """This is the main function!"""
        dir_path = 'F:\\download\\Test_folder1'
        ftp.mkdir(dir_path)
        os.chdir(dir_path)

        """
        # If the address is complete, simply call this function to implement the download 
        root_url = 'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/README'
        ftp.static_download(root_url)
        """

        # If the address is incomplete, simply call this function to implement the download
        root_url = 'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/'
        txt = Ftp.static_download(root_url)
        Ftp.static_readTxt(txt, root_url)

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ftp = Ftp()
    ftp.main()
